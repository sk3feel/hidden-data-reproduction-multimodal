from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from anonymize import (
    find_all_answer_spans,
    find_answer_span,
    mask_image,
    span_bboxes_from_spans,
)
from label_docvqa_gigachat import load_docvqa_examples
from load_data import extract_ocr_tokens

DEFAULT_LABELS_CSV = (
    Path("artifacts")
    / "field_labeling"
    / "merged"
    / "pixparse__docvqa-single-page-questions__all_splits.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts") / "docqa_recovery" / "benchmark"

COARSE_FIELD_TYPE_MAP = {
    "DATE_TIME": "DATE",
    "MONEY": "AMOUNT",
    "QUANTITY": "AMOUNT",
    "PERCENTAGE": "AMOUNT",
    "IDENTIFIER": "ID",
    "DOCUMENT_REFERENCE": "ID",
    "PERSON_NAME": "PERSON",
    "ORG_NAME": "ORG",
    "ADDRESS": "CONTACT_ADR",
    "CONTACT": "CONTACT_ADR",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ocr_strategy: str
    image_strategy: str
    context_window: int
    blur_sigma: float | None = None


def _image_scenario_id(image_strategy: str, blur_sigma: float | None = None) -> str:
    if image_strategy != "blur":
        return image_strategy
    if blur_sigma is None:
        return "blur"
    sigma_value = int(blur_sigma) if float(blur_sigma).is_integer() else blur_sigma
    return f"blur_{sigma_value}"


def default_scenarios() -> list[Scenario]:
    scenarios = [Scenario("original", "none", "none", 0)]
    ocr_strategies = ["none", "drop", "mask"]
    image_variants = [
        ("none", None),
        ("black", None),
        ("white", None),
        ("blur", 10.0),
        ("blur", 20.0),
        ("blur", 50.0),
    ]
    context_windows = [0, 20]

    for context_window in context_windows:
        for ocr_strategy in ocr_strategies:
            for image_strategy, blur_sigma in image_variants:
                if ocr_strategy == "none" and image_strategy == "none":
                    continue
                image_id = _image_scenario_id(
                    image_strategy=image_strategy,
                    blur_sigma=blur_sigma,
                )
                scenarios.append(
                    Scenario(
                        scenario_id=f"ocr_{ocr_strategy}__img_{image_id}__k_{context_window}",
                        ocr_strategy=ocr_strategy,
                        image_strategy=image_strategy,
                        context_window=context_window,
                        blur_sigma=blur_sigma,
                    )
                )
    return scenarios


def _to_pil_image(image_obj: Any) -> Image.Image | None:
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert("RGB")
    return Image.fromarray(image_obj).convert("RGB")


def _normalize_bbox(bbox: Any) -> Any:
    if isinstance(bbox, tuple):
        return [_normalize_bbox(x) for x in bbox]
    if isinstance(bbox, list):
        return [_normalize_bbox(x) for x in bbox]
    if isinstance(bbox, dict):
        return {str(k): _normalize_bbox(v) for k, v in bbox.items()}
    return bbox


def _normalize_token_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(entry.get("text", "")),
        "bbox": _normalize_bbox(entry.get("bbox")),
    }


def _expand_span(
    start_idx: int,
    end_idx: int,
    num_tokens: int,
    context_window: int,
) -> tuple[int, int]:
    if num_tokens <= 0:
        return start_idx, end_idx
    return (
        max(0, start_idx - context_window),
        min(num_tokens - 1, end_idx + context_window),
    )


def _record_spans(record: dict[str, Any]) -> list[tuple[int, int]]:
    raw_spans = record.get("answer_spans")
    if isinstance(raw_spans, list) and raw_spans:
        spans: list[tuple[int, int]] = []
        for item in raw_spans:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                spans.append((int(item[0]), int(item[1])))
        if spans:
            return spans
    return [
        (
            int(record["answer_start_idx"]),
            int(record["answer_end_idx"]),
        )
    ]


def _expanded_spans(
    spans: list[tuple[int, int]],
    num_tokens: int,
    context_window: int,
) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start_idx, end_idx in spans:
        span = _expand_span(
            start_idx=start_idx,
            end_idx=end_idx,
            num_tokens=num_tokens,
            context_window=context_window,
        )
        if span not in seen:
            expanded.append(span)
            seen.add(span)
    return expanded


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _load_labels_rows(
    labels_csv: Path,
    splits: set[str] | None,
    coarse_types: set[str] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip()
            field_type = str(row.get("field_type", "")).strip()
            coarse_type = COARSE_FIELD_TYPE_MAP.get(field_type)
            if splits and split not in splits:
                continue
            if coarse_types and coarse_type not in coarse_types:
                continue
            if not coarse_type:
                continue
            answer = str(row.get("answer", "")).strip()
            question = str(row.get("question", "")).strip()
            if not answer or not question:
                continue
            copied = dict(row)
            copied["coarse_field_type"] = coarse_type
            rows.append(copied)
    return rows


def _stratified_sample(
    rows: list[dict[str, str]],
    max_examples: int | None,
    seed: int,
) -> list[dict[str, str]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["coarse_field_type"], []).append(row)

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    selected: list[dict[str, str]] = []
    coarse_types = sorted(grouped)
    base = max_examples // len(coarse_types)
    remainder = max_examples % len(coarse_types)

    for idx, coarse_type in enumerate(coarse_types):
        take = base + (1 if idx < remainder else 0)
        selected.extend(grouped[coarse_type][:take])

    return selected[:max_examples]


def _index_examples_by_local_row_id(
    examples: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {idx: example for idx, example in enumerate(examples)}


def build_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    labels_csv: str | Path = DEFAULT_LABELS_CSV,
    splits: list[str] | None = None,
    coarse_types: list[str] | None = None,
    max_examples: int | None = None,
    allow_network: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    labels_csv = Path(labels_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = _load_labels_rows(
        labels_csv=labels_csv,
        splits=set(splits or []),
        coarse_types=set(coarse_types or []),
    )
    selected_rows = _stratified_sample(selected_rows, max_examples=max_examples, seed=seed)
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for row in selected_rows:
        rows_by_split.setdefault(row["split"], []).append(row)

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    images_dir = output_dir / "images" / "original"

    total_seen = 0
    total_kept = 0
    skipped_no_span = 0
    skipped_no_image = 0
    skipped_no_bbox = 0
    type_counts: dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for split, split_rows in sorted(rows_by_split.items()):
            examples = load_docvqa_examples(
                split=split,
                limit=None,
                allow_network=allow_network,
            )
            example_index = _index_examples_by_local_row_id(examples)

            for row in split_rows:
                total_seen += 1
                local_row_id = int(row["local_row_id"])
                example = example_index.get(local_row_id)
                if example is None:
                    continue

                image = _to_pil_image(example.get("image"))
                if image is None:
                    skipped_no_image += 1
                    continue

                token_entries = [
                    _normalize_token_entry(entry) for entry in extract_ocr_tokens(example)
                ]
                ocr_texts = [entry["text"] for entry in token_entries]
                answer = str(row["answer"]).strip()
                raw_answers = example.get("answers") or [answer]
                answer_candidates: list[str] = []
                seen_answers: set[str] = set()
                for candidate in [answer, *raw_answers]:
                    text = str(candidate).strip()
                    if not text or text in seen_answers:
                        continue
                    answer_candidates.append(text)
                    seen_answers.add(text)

                matched_spans: list[tuple[int, int]] = []
                seen_spans: set[tuple[int, int]] = set()
                for candidate in answer_candidates:
                    for span in find_all_answer_spans(ocr_texts, candidate):
                        if span in seen_spans:
                            continue
                        matched_spans.append(span)
                        seen_spans.add(span)

                found = bool(matched_spans)
                start_idx, end_idx, _ = find_answer_span(ocr_texts, answer)
                if (start_idx is None or end_idx is None) and matched_spans:
                    start_idx, end_idx = matched_spans[0]
                if not found or start_idx is None or end_idx is None:
                    skipped_no_span += 1
                    continue

                answer_bboxes = span_bboxes_from_spans(token_entries, matched_spans)
                if not answer_bboxes:
                    skipped_no_bbox += 1
                    continue
                answer_bbox = answer_bboxes[0]

                example_id = str(row.get("example_id") or f"{split}_{local_row_id}")
                image_path = images_dir / f"{example_id}.png"
                if not image_path.exists():
                    _save_image(image, image_path)

                manifest_row = {
                    "dataset_name": row.get("dataset_name", ""),
                    "split": split,
                    "local_row_id": local_row_id,
                    "example_id": example_id,
                    "question": str(row["question"]).strip(),
                    "answer": answer,
                    "answers": answer_candidates,
                    "field_type": str(row["field_type"]).strip(),
                    "coarse_field_type": str(row["coarse_field_type"]).strip(),
                    "field_group": str(row.get("field_group", "")).strip(),
                    "sensitivity": str(row.get("sensitivity", "")).strip(),
                    "answer_start_idx": start_idx,
                    "answer_end_idx": end_idx,
                    "answer_bbox": list(answer_bbox),
                    "answer_spans": [[span_start, span_end] for span_start, span_end in matched_spans],
                    "answer_bboxes": [list(bbox) for bbox in answer_bboxes],
                    "ocr_tokens": token_entries,
                    "image_path": str(image_path.resolve()),
                    "image_size": list(image.size),
                }
                manifest_file.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
                total_kept += 1
                coarse_type = manifest_row["coarse_field_type"]
                type_counts[coarse_type] = type_counts.get(coarse_type, 0) + 1

    summary = {
        "manifest_path": str(manifest_path.resolve()),
        "total_seen": total_seen,
        "total_kept": total_kept,
        "skipped_no_span": skipped_no_span,
        "skipped_no_image": skipped_no_image,
        "skipped_no_bbox": skipped_no_bbox,
        "type_counts": type_counts,
        "scenarios": [asdict(scenario) for scenario in default_scenarios()],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def load_benchmark_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scenarios(scenarios_path: str | Path | None = None) -> list[Scenario]:
    if scenarios_path is None:
        return default_scenarios()
    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    return [Scenario(**item) for item in payload]


def render_image_variant(
    image_path: str | Path,
    token_entries: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    image_strategy: str,
    context_window: int,
    blur_sigma: float | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image_strategy == "none":
        return image

    expanded_spans = _expanded_spans(
        spans=spans,
        num_tokens=len(token_entries),
        context_window=context_window,
    )
    bboxes = span_bboxes_from_spans(token_entries, expanded_spans)
    if not bboxes:
        return image

    out = image.copy()
    for bbox in bboxes:
        out = mask_image(
            image=out,
            bbox=bbox,
            strategy=image_strategy,
            blur_sigma=12.0 if blur_sigma is None else blur_sigma,
        )
    return out


def redact_ocr_tokens_with_context(
    token_entries: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    ocr_strategy: str,
    context_window: int,
) -> list[dict[str, Any]]:
    if ocr_strategy == "none":
        return [dict(entry) for entry in token_entries]

    expanded_spans = _expanded_spans(
        spans=spans,
        num_tokens=len(token_entries),
        context_window=context_window,
    )
    masked_indices: set[int] = set()
    for context_start, context_end in expanded_spans:
        masked_indices.update(range(context_start, context_end + 1))

    if ocr_strategy == "drop":
        return [
            dict(entry)
            for idx, entry in enumerate(token_entries)
            if idx not in masked_indices
        ]

    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(token_entries):
        copied = dict(entry)
        if idx in masked_indices:
            copied["text"] = "[REDACTED]"
        out.append(copied)
    return out


def scenario_payload(record: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    token_entries = record["ocr_tokens"]
    spans = _record_spans(record)
    return {
        "scenario_id": scenario.scenario_id,
        "question": record["question"],
        "gold_answer": record["answer"],
        "answers": record.get("answers") or [record["answer"]],
        "field_type": record["field_type"],
        "coarse_field_type": record["coarse_field_type"],
        "image": render_image_variant(
            image_path=record["image_path"],
            token_entries=token_entries,
            spans=spans,
            image_strategy=scenario.image_strategy,
            context_window=scenario.context_window,
            blur_sigma=scenario.blur_sigma,
        ),
        "ocr_tokens": redact_ocr_tokens_with_context(
            token_entries=token_entries,
            spans=spans,
            ocr_strategy=scenario.ocr_strategy,
            context_window=scenario.context_window,
        ),
        "image_path": record["image_path"],
        "original_record": record,
    }
