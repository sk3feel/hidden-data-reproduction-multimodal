from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFile, ImageFilter

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_BENCHMARK_DIR = ROOT / "artifacts" / "docqa_recovery" / "benchmark"
DEFAULT_REPORT_PATH = (
    ROOT / "artifacts" / "docqa_recovery" / "validation" / "anonymization_audit_report.json"
)


@dataclass
class ScenarioAudit:
    scenario_id: str
    ocr_strategy: str
    image_strategy: str
    context_window: int
    total_records: int
    ocr_checked: int = 0
    ocr_leaks: int = 0
    image_checked: int = 0
    image_unchanged: int = 0
    example_failures: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "ocr_strategy": self.ocr_strategy,
            "image_strategy": self.image_strategy,
            "context_window": self.context_window,
            "total_records": self.total_records,
            "ocr_checked": self.ocr_checked,
            "ocr_leaks": self.ocr_leaks,
            "image_checked": self.image_checked,
            "image_unchanged": self.image_unchanged,
            "example_failures": self.example_failures or [],
        }


def crop_diff_exists(original: Image.Image, masked: Image.Image, bbox: list[int]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = max(left + 1, max(x1, x2))
    bottom = max(top + 1, max(y1, y2))
    original_crop = original.crop((left, top, right, bottom))
    masked_crop = masked.crop((left, top, right, bottom))
    return original_crop.tobytes() != masked_crop.tobytes()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _flatten_ocr_tokens(ocr_tokens: list[str]) -> tuple[list[str], list[int]]:
    flat_tokens: list[str] = []
    flat_to_orig_idx: list[int] = []
    for orig_idx, token in enumerate(ocr_tokens):
        for part in normalize_text(token).split():
            flat_tokens.append(part)
            flat_to_orig_idx.append(orig_idx)
    return flat_tokens, flat_to_orig_idx


def find_all_answer_spans(ocr_tokens: list[str], answer: str) -> list[tuple[int, int]]:
    answer_parts = normalize_text(answer).split()
    if not answer_parts:
        return []

    flat_tokens, flat_to_orig_idx = _flatten_ocr_tokens(ocr_tokens)
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    n = len(answer_parts)
    for start in range(len(flat_tokens) - n + 1):
        if flat_tokens[start : start + n] != answer_parts:
            continue
        span = (flat_to_orig_idx[start], flat_to_orig_idx[start + n - 1])
        if span not in seen:
            spans.append(span)
            seen.add(span)
    return spans


def load_benchmark_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ocr_strategy: str
    image_strategy: str
    context_window: int


def load_scenarios() -> list[Scenario]:
    scenarios = [Scenario("original", "none", "none", 0)]
    for context_window in [0, 20]:
        for ocr_strategy in ["none", "drop", "mask"]:
            for image_strategy in ["none", "black", "white", "blur"]:
                if ocr_strategy == "none" and image_strategy == "none":
                    continue
                scenarios.append(
                    Scenario(
                        scenario_id=f"ocr_{ocr_strategy}__img_{image_strategy}__k_{context_window}",
                        ocr_strategy=ocr_strategy,
                        image_strategy=image_strategy,
                        context_window=context_window,
                    )
                )
    return scenarios


def _bbox_to_xyxy(
    bbox: list[int] | list[float] | tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if max(x1, y1, x2, y2) <= 1.0:
        width, height = image_size
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return tuple(int(round(v)) for v in [x1, y1, x2, y2])


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
            if isinstance(item, list) and len(item) == 2:
                spans.append((int(item[0]), int(item[1])))
        if spans:
            return spans
    return [(int(record["answer_start_idx"]), int(record["answer_end_idx"]))]


def _expanded_spans(
    spans: list[tuple[int, int]],
    num_tokens: int,
    context_window: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start_idx, end_idx in spans:
        span = _expand_span(start_idx, end_idx, num_tokens, context_window)
        if span not in seen:
            out.append(span)
            seen.add(span)
    return out


def render_image_variant(
    image_path: str | Path,
    token_entries: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    image_strategy: str,
    context_window: int,
) -> Image.Image:
    with Image.open(image_path) as image_obj:
        image = image_obj.convert("RGB")
    if image_strategy == "none":
        return image

    expanded_spans = _expanded_spans(spans, len(token_entries), context_window)
    bboxes: list[list[int]] = []
    for start_idx, end_idx in expanded_spans:
        for token in token_entries[start_idx : end_idx + 1]:
            bbox = token.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                bboxes.append([int(v) for v in bbox])
    if not bboxes:
        return image

    merged_boxes: list[tuple[int, int, int, int]] = []
    for bbox in bboxes:
        merged_boxes.append(_bbox_to_xyxy(bbox, image.size))

    if image_strategy == "blur":
        out = image.copy()
        for x1, y1, x2, y2 in merged_boxes:
            crop = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=12))
            out.paste(crop, (x1, y1, x2, y2))
        return out

    out = image.copy()
    draw = ImageDraw.Draw(out)
    fill = "black" if image_strategy == "black" else "white"
    for x1, y1, x2, y2 in merged_boxes:
        draw.rectangle([x1, y1, x2, y2], fill=fill)
    return out


def redact_ocr_tokens_with_context(
    token_entries: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    ocr_strategy: str,
    context_window: int,
) -> list[dict[str, Any]]:
    if ocr_strategy == "none":
        return [dict(entry) for entry in token_entries]

    expanded_spans = _expanded_spans(spans, len(token_entries), context_window)
    masked_indices: set[int] = set()
    for context_start, context_end in expanded_spans:
        masked_indices.update(range(context_start, context_end + 1))

    if ocr_strategy == "drop":
        return [dict(entry) for idx, entry in enumerate(token_entries) if idx not in masked_indices]

    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(token_entries):
        copied = dict(entry)
        if idx in masked_indices:
            copied["text"] = "[REDACTED]"
        out.append(copied)
    return out


def scenario_payload(record: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    token_entries = [dict(entry) for entry in record["ocr_tokens"]]
    spans = _record_spans(record)
    return {
        "scenario_id": scenario.scenario_id,
        "ocr_tokens": redact_ocr_tokens_with_context(
            token_entries=token_entries,
            spans=spans,
            ocr_strategy=scenario.ocr_strategy,
            context_window=scenario.context_window,
        ),
        "image": render_image_variant(
            image_path=record["image_path"],
            token_entries=token_entries,
            spans=spans,
            image_strategy=scenario.image_strategy,
            context_window=scenario.context_window,
        ),
    }


def answer_variants(record: dict[str, Any]) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()
    for candidate in record.get("answers") or [record.get("answer", "")]:
        text = str(candidate).strip()
        if not text or text in seen:
            continue
        variants.append(text)
        seen.add(text)
    return variants


def answer_bboxes(record: dict[str, Any]) -> list[list[int]]:
    bboxes = record.get("answer_bboxes")
    if isinstance(bboxes, list) and bboxes:
        out: list[list[int]] = []
        for bbox in bboxes:
            if isinstance(bbox, list) and len(bbox) == 4:
                out.append([int(v) for v in bbox])
        if out:
            return out

    bbox = record.get("answer_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [[int(v) for v in bbox]]
    return []


def audit_scenario(
    records: list[dict[str, Any]],
    scenario: Any,
    max_failure_examples: int,
) -> ScenarioAudit:
    summary = ScenarioAudit(
        scenario_id=str(scenario.scenario_id),
        ocr_strategy=str(scenario.ocr_strategy),
        image_strategy=str(scenario.image_strategy),
        context_window=int(scenario.context_window),
        total_records=len(records),
        example_failures=[],
    )

    for record in records:
        payload = scenario_payload(record, scenario)
        failure: dict[str, Any] = {
            "example_id": str(record["example_id"]),
            "ocr_leaked_answers": [],
            "unchanged_bboxes": [],
        }

        if scenario.ocr_strategy != "none":
            summary.ocr_checked += 1
            ocr_texts = [str(token.get("text", "")) for token in payload["ocr_tokens"]]
            leaked_answers: list[str] = []
            for answer in answer_variants(record):
                if find_all_answer_spans(ocr_texts, answer):
                    leaked_answers.append(answer)
            if leaked_answers:
                summary.ocr_leaks += 1
                failure["ocr_leaked_answers"] = leaked_answers

        if scenario.image_strategy != "none":
            summary.image_checked += 1
            unchanged_bboxes: list[list[int]] = []
            with Image.open(record["image_path"]) as original_image:
                original = original_image.convert("RGB")
            masked = payload["image"]
            for bbox in answer_bboxes(record):
                if not crop_diff_exists(original, masked, bbox):
                    unchanged_bboxes.append(bbox)
            if unchanged_bboxes:
                summary.image_unchanged += 1
                failure["unchanged_bboxes"] = unchanged_bboxes

        if (
            failure["ocr_leaked_answers"] or failure["unchanged_bboxes"]
        ) and len(summary.example_failures or []) < max_failure_examples:
            (summary.example_failures or []).append(failure)

    return summary


def audit_benchmark(
    benchmark_dir: Path,
    report_path: Path,
    max_failure_examples: int,
) -> dict[str, Any]:
    manifest_path = benchmark_dir / "manifest.jsonl"
    records = load_benchmark_manifest(manifest_path)
    scenarios = [
        scenario
        for scenario in load_scenarios()
        if scenario.ocr_strategy != "none" or scenario.image_strategy != "none"
    ]

    scenario_summaries = [
        audit_scenario(
            records=records,
            scenario=scenario,
            max_failure_examples=max_failure_examples,
        )
        for scenario in scenarios
    ]

    report = {
        "benchmark_dir": str(benchmark_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "num_records": len(records),
        "num_scenarios_checked": len(scenarios),
        "all_passed": all(
            summary.ocr_leaks == 0 and summary.image_unchanged == 0
            for summary in scenario_summaries
        ),
        "scenario_results": [summary.to_dict() for summary in scenario_summaries],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full audit for benchmark anonymization.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=str(DEFAULT_BENCHMARK_DIR),
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
    )
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=20,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_benchmark(
        benchmark_dir=Path(args.benchmark_dir),
        report_path=Path(args.report_path),
        max_failure_examples=max(args.max_failure_examples, 1),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
