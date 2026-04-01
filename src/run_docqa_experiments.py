from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from docqa_benchmark import (
    DEFAULT_OUTPUT_DIR,
    build_benchmark,
    load_benchmark_manifest,
    load_scenarios,
    scenario_payload,
)
from docqa_metrics import (
    best_metric_over_answers,
    build_answer_pool,
    estimate_random_baseline,
)


@dataclass
class ModelRequest:
    model_kind: str
    model_name: str
    mode: str
    device: str
    quantize_4bit: bool
    max_new_tokens: int


class BaseRunner:
    def predict(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError


class OracleRunner(BaseRunner):
    def predict(self, payload: dict[str, Any]) -> str:
        return str(payload["gold_answer"])


class RandomBaselineRunner(BaseRunner):
    def __init__(self, answer_pool: dict[str, list[dict[str, Any]]], seed: int) -> None:
        self.answer_pool = answer_pool
        self.seed = seed

    def predict(self, payload: dict[str, Any]) -> str:
        record = payload["original_record"]
        candidates = list(self.answer_pool.get(record["coarse_field_type"], []))
        if not candidates:
            return ""
        filtered = [
            candidate
            for candidate in candidates
            if str(candidate.get("example_id")) != str(record.get("example_id"))
        ]
        if filtered:
            candidates = filtered
        rng = random.Random(
            self.seed
            + int(record.get("local_row_id", 0))
            + sum(ord(ch) for ch in str(record.get("coarse_field_type", "")))
        )
        return str(rng.choice(candidates).get("answer", "")).strip()


class LayoutLMv3Runner(BaseRunner):
    def __init__(self, request: ModelRequest) -> None:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForDocumentQuestionAnswering,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LayoutLMv3 runner requires torch and transformers."
            ) from exc

        self.torch = torch
        self.config = AutoConfig.from_pretrained(request.model_name)
        self.model = AutoModelForDocumentQuestionAnswering.from_pretrained(request.model_name)
        self.model.to(request.device)
        self.model.eval()
        self.mode = request.mode
        self.model_family = str(getattr(self.config, "model_type", ""))

        if self.model_family == "layoutlm":
            self.tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(request.model_name, apply_ocr=False)
            self.tokenizer = getattr(self.processor, "tokenizer", None)

    @staticmethod
    def _normalize_boxes(
        ocr_tokens: list[dict[str, Any]],
        image_size: tuple[int, int],
    ) -> list[list[int]]:
        width, height = image_size
        normalized: list[list[int]] = []
        for token in ocr_tokens:
            bbox = token.get("bbox")
            if bbox is None:
                normalized.append([0, 0, 0, 0])
                continue
            from anonymize import _bbox_to_xyxy

            x1, y1, x2, y2 = _bbox_to_xyxy(bbox, image_size=image_size)
            normalized.append(
                [
                    int(max(0, min(1000, round(1000 * x1 / max(width, 1))))),
                    int(max(0, min(1000, round(1000 * y1 / max(height, 1))))),
                    int(max(0, min(1000, round(1000 * x2 / max(width, 1))))),
                    int(max(0, min(1000, round(1000 * y2 / max(height, 1))))),
                ]
            )
        return normalized

    def predict(self, payload: dict[str, Any]) -> str:
        image = payload["image"]
        words = [token["text"] for token in payload["ocr_tokens"]]
        boxes = self._normalize_boxes(payload["ocr_tokens"], image.size)
        text_only_mode = self.mode in {"ocr_only", "text_only"}
        if self.model_family == "layoutlm":
            question_words = str(payload["question"]).split()
            encoding = self.tokenizer(
                question_words,
                words,
                is_split_into_words=True,
                truncation=True,
                return_tensors="pt",
            )

            sequence_ids = encoding.sequence_ids(0)
            word_ids = encoding.word_ids(0)
            token_boxes: list[list[int]] = []
            for sequence_id, word_id in zip(sequence_ids, word_ids):
                if sequence_id != 1 or word_id is None or word_id >= len(boxes):
                    token_boxes.append([0, 0, 0, 0])
                    continue
                token_boxes.append(boxes[word_id])

            encoding["bbox"] = self.torch.tensor(
                [token_boxes],
                dtype=self.torch.long,
            )
        else:
            image = payload["image"]
            if text_only_mode:
                from PIL import Image

                image = Image.new("RGB", image.size, color="white")

            encoding = self.processor(
                images=image,
                text=[payload["question"]],
                text_pair=[words],
                boxes=[boxes],
                truncation=True,
                return_tensors="pt",
            )
        encoding = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in encoding.items()
        }

        with self.torch.no_grad():
            outputs = self.model(**encoding)

        start_idx = int(outputs.start_logits.argmax(-1).item())
        end_idx = int(outputs.end_logits.argmax(-1).item())
        if end_idx < start_idx:
            end_idx = start_idx

        input_ids = encoding["input_ids"].squeeze(0)
        span_ids = input_ids[start_idx : end_idx + 1]
        tokenizer = self.tokenizer or self.processor.tokenizer
        decoded = tokenizer.decode(span_ids, skip_special_tokens=True)
        return decoded.strip()


class DonutRunner(BaseRunner):
    def __init__(self, request: ModelRequest) -> None:
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Donut runner requires torch, transformers, sentencepiece, and protobuf."
            ) from exc

        self.torch = torch
        try:
            self.processor = DonutProcessor.from_pretrained(request.model_name)
        except (ImportError, ValueError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to load DonutProcessor. Install sentencepiece and protobuf in the active environment."
            ) from exc
        self.model = VisionEncoderDecoderModel.from_pretrained(request.model_name)
        self.model.to(request.device)
        self.model.eval()
        self.max_new_tokens = request.max_new_tokens

    def predict(self, payload: dict[str, Any]) -> str:
        prompt = (
            f"<s_docvqa><s_question>{payload['question']}</s_question><s_answer>"
        )
        decoder_input_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.model.device)
        pixel_values = self.processor(
            payload["image"],
            return_tensors="pt",
        ).pixel_values.to(self.model.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        if "<s_answer>" in decoded:
            decoded = decoded.split("<s_answer>", 1)[1]
        if "</s_answer>" in decoded:
            decoded = decoded.split("</s_answer>", 1)[0]
        decoded = decoded.replace(self.processor.tokenizer.eos_token or "", "")
        decoded = decoded.replace(self.processor.tokenizer.pad_token or "", "")
        return decoded.strip()


class Qwen2VLRunner(BaseRunner):
    def __init__(self, request: ModelRequest) -> None:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Qwen2-VL runner requires torch and transformers.") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(request.model_name)
        model_kwargs: dict[str, Any] = {}
        if request.quantize_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = "auto"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            request.model_name,
            **model_kwargs,
        )
        if not request.quantize_4bit:
            self.model.to(request.device)
        self.model.eval()
        self.max_new_tokens = request.max_new_tokens

    def predict(self, payload: dict[str, Any]) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": payload["image"]},
                    {
                        "type": "text",
                        "text": (
                            "Answer the document question with a short span copied from the "
                            f"document when possible.\nQuestion: {payload['question']}"
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=[payload["image"]],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, prompt_len:]
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decoded.strip()


def build_runner(
    request: ModelRequest,
    answer_pool: dict[str, list[dict[str, Any]]],
    seed: int,
) -> BaseRunner:
    if request.model_kind == "oracle":
        return OracleRunner()
    if request.model_kind == "random_baseline":
        return RandomBaselineRunner(answer_pool=answer_pool, seed=seed)
    if request.model_kind == "layoutlmv3":
        return LayoutLMv3Runner(request)
    if request.model_kind == "donut":
        return DonutRunner(request)
    if request.model_kind == "qwen2_vl":
        return Qwen2VLRunner(request)
    raise ValueError(f"Unsupported model kind: {request.model_kind}")


def aggregate_metrics(predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions_df.empty:
        empty_overall = pd.DataFrame(
            columns=[
                "model_kind",
                "model_name",
                "mode",
                "scenario_id",
                "n_examples",
                "exact_match",
                "token_f1",
                "random_em",
                "random_f1",
                "corrected_em",
                "corrected_f1",
            ]
        )
        empty_by_type = pd.DataFrame(
            columns=[
                "model_kind",
                "model_name",
                "mode",
                "scenario_id",
                "coarse_field_type",
                "n_examples",
                "exact_match",
                "token_f1",
                "random_em",
                "random_f1",
                "corrected_em",
                "corrected_f1",
            ]
        )
        return empty_overall, empty_by_type

    overall = (
        predictions_df.groupby(["model_kind", "model_name", "mode", "scenario_id"], dropna=False)
        .agg(
            n_examples=("example_id", "count"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            random_em=("random_em", "mean"),
            random_f1=("random_f1", "mean"),
            corrected_em=("corrected_em", "mean"),
            corrected_f1=("corrected_f1", "mean"),
        )
        .reset_index()
        .sort_values(["model_kind", "mode", "scenario_id"])
    )

    by_type = (
        predictions_df.groupby(
            ["model_kind", "model_name", "mode", "scenario_id", "coarse_field_type"],
            dropna=False,
        )
        .agg(
            n_examples=("example_id", "count"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            random_em=("random_em", "mean"),
            random_f1=("random_f1", "mean"),
            corrected_em=("corrected_em", "mean"),
            corrected_f1=("corrected_f1", "mean"),
        )
        .reset_index()
        .sort_values(["model_kind", "mode", "scenario_id", "coarse_field_type"])
    )
    return overall, by_type


def stratified_limit_records(
    records: list[dict[str, Any]],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or len(records) <= limit:
        return records

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["coarse_field_type"]), []).append(record)

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    coarse_types = sorted(grouped)
    base = limit // len(coarse_types)
    remainder = limit % len(coarse_types)

    for idx, coarse_type in enumerate(coarse_types):
        take = base + (1 if idx < remainder else 0)
        selected.extend(grouped[coarse_type][:take])

    if len(selected) < limit:
        leftovers: list[dict[str, Any]] = []
        for coarse_type in coarse_types:
            leftovers.extend(grouped[coarse_type][base + 1 :])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])

    rng.shuffle(selected)
    return selected[:limit]


def run_predictions(
    manifest_path: str | Path,
    output_dir: str | Path,
    request: ModelRequest,
    scenarios_path: str | Path | None = None,
    seed: int = 42,
    limit: int | None = None,
    random_baseline_samples: int = 20,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_benchmark_manifest(manifest_path)
    records = stratified_limit_records(records=records, limit=limit, seed=seed)
    scenarios = load_scenarios(scenarios_path)
    answer_pool = build_answer_pool(records)
    runner = build_runner(request=request, answer_pool=answer_pool, seed=seed)

    predictions: list[dict[str, Any]] = []
    records_iter = tqdm(
        records,
        total=len(records),
        desc=f"{request.model_kind}:{request.mode}",
    )
    for record in records_iter:
        random_em, random_f1 = estimate_random_baseline(
            record=record,
            answer_pool=answer_pool,
            num_samples=random_baseline_samples,
            seed=seed,
        )
        for scenario in scenarios:
            payload = scenario_payload(record, scenario)
            prediction = runner.predict(payload)
            exact_match, token_f1 = best_metric_over_answers(
                prediction=prediction,
                answers=list(record.get("answers") or [record["answer"]]),
            )
            predictions.append(
                {
                    "model_kind": request.model_kind,
                    "model_name": request.model_name,
                    "mode": request.mode,
                    "scenario_id": scenario.scenario_id,
                    "split": record["split"],
                    "example_id": record["example_id"],
                    "local_row_id": record["local_row_id"],
                    "field_type": record["field_type"],
                    "coarse_field_type": record["coarse_field_type"],
                    "question": record["question"],
                    "gold_answer": record["answer"],
                    "prediction": prediction,
                    "exact_match": exact_match,
                    "token_f1": token_f1,
                    "random_em": random_em,
                    "random_f1": random_f1,
                    "corrected_em": exact_match - random_em,
                    "corrected_f1": token_f1 - random_f1,
                }
            )

    predictions_df = pd.DataFrame(predictions)
    overall_df, by_type_df = aggregate_metrics(predictions_df)

    prefix = f"{request.model_kind}__{request.mode}"
    predictions_path = output_dir / f"{prefix}__predictions.csv"
    overall_path = output_dir / f"{prefix}__overall_metrics.csv"
    by_type_path = output_dir / f"{prefix}__metrics_by_type.csv"
    config_path = output_dir / f"{prefix}__config.json"

    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8")
    by_type_df.to_csv(by_type_path, index=False, encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "manifest_path": str(Path(manifest_path).resolve()),
                "model_request": request.__dict__,
                "seed": seed,
                "random_baseline_samples": random_baseline_samples,
                "num_records": len(records),
                "num_scenarios": len(scenarios),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "predictions_path": str(predictions_path.resolve()),
        "overall_metrics_path": str(overall_path.resolve()),
        "metrics_by_type_path": str(by_type_path.resolve()),
        "num_predictions": len(predictions_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run DocQA recovery experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-benchmark")
    prepare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare_parser.add_argument(
        "--labels-csv",
        default="artifacts/field_labeling/merged/pixparse__docvqa-single-page-questions__all_splits.csv",
    )
    prepare_parser.add_argument("--splits", nargs="+", default=["validation"])
    prepare_parser.add_argument(
        "--coarse-types",
        nargs="+",
        default=["DATE", "AMOUNT", "ID", "PERSON", "ORG", "CONTACT_ADR"],
        help="Coarse field types to include in the benchmark",
    )
    prepare_parser.add_argument("--max-examples", type=int, default=None)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--allow-network", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_OUTPUT_DIR / "manifest.jsonl"),
    )
    run_parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "docqa_recovery" / "runs"),
    )
    run_parser.add_argument(
        "--model-kind",
        choices=["oracle", "random_baseline", "layoutlmv3", "donut", "qwen2_vl"],
        required=True,
    )
    run_parser.add_argument("--model-name", default="")
    run_parser.add_argument(
        "--mode",
        default="multimodal",
        choices=["multimodal", "ocr_only", "text_only", "image_only"],
    )
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--quantize-4bit", action="store_true")
    run_parser.add_argument("--max-new-tokens", type=int, default=64)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--random-baseline-samples", type=int, default=20)
    run_parser.add_argument("--scenarios-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-benchmark":
        summary = build_benchmark(
            output_dir=args.output_dir,
            labels_csv=args.labels_csv,
            splits=args.splits,
            coarse_types=args.coarse_types,
            max_examples=args.max_examples,
            allow_network=args.allow_network,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    model_name = args.model_name.strip()
    if not model_name:
        defaults = {
            "oracle": "oracle",
            "random_baseline": "random_baseline",
            "layoutlmv3": "impira/layoutlm-document-qa",
            "donut": "naver-clova-ix/donut-base-finetuned-docvqa",
            "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct",
        }
        model_name = defaults[args.model_kind]

    request = ModelRequest(
        model_kind=args.model_kind,
        model_name=model_name,
        mode=args.mode,
        device=args.device,
        quantize_4bit=bool(args.quantize_4bit),
        max_new_tokens=args.max_new_tokens,
    )
    summary = run_predictions(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        request=request,
        scenarios_path=args.scenarios_path,
        seed=args.seed,
        limit=args.limit,
        random_baseline_samples=args.random_baseline_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
