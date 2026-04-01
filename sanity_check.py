from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from docqa_benchmark import load_benchmark_manifest, load_scenarios, scenario_payload


BENCHMARK_DIR = ROOT / "artifacts" / "docqa_recovery" / "benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.jsonl"
SUMMARY_PATH = BENCHMARK_DIR / "summary.json"
PREVIEW_DIR = ROOT / "artifacts" / "docqa_recovery" / "sanity_check_preview"
SCENARIO_PREVIEW_DIR = PREVIEW_DIR / "scenario_smoke"

EXPECTED_COLUMNS = {
    "dataset_name",
    "split",
    "local_row_id",
    "example_id",
    "question",
    "answer",
    "answers",
    "field_type",
    "coarse_field_type",
    "field_group",
    "sensitivity",
    "answer_start_idx",
    "answer_end_idx",
    "answer_bbox",
    "ocr_tokens",
    "image_path",
    "image_size",
}
NON_EMPTY_COLUMNS = {
    "split",
    "example_id",
    "question",
    "answer",
    "field_type",
    "coarse_field_type",
    "image_path",
}
MASK_TOKEN = "[REDACTED]"
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def add_result(results: list[CheckResult], name: str, passed: bool, details: str) -> None:
    results.append(CheckResult(name=name, passed=passed, details=details))


def crop_diff_exists(original: Image.Image, masked: Image.Image, bbox: list[int]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = max(left + 1, max(x1, x2))
    bottom = max(top + 1, max(y1, y2))
    original_crop = original.crop((left, top, right, bottom))
    masked_crop = masked.crop((left, top, right, bottom))
    return original_crop.tobytes() != masked_crop.tobytes()


def main() -> int:
    results: list[CheckResult] = []

    benchmark_exists = BENCHMARK_DIR.exists() and any(BENCHMARK_DIR.iterdir())
    add_result(
        results,
        "benchmark_dir_non_empty",
        benchmark_exists,
        f"dir={BENCHMARK_DIR}",
    )
    if not benchmark_exists:
        return finalize(results)

    manifest_exists = MANIFEST_PATH.exists()
    add_result(
        results,
        "manifest_jsonl_exists",
        manifest_exists,
        (
            f"found {MANIFEST_PATH.name}; the pipeline stores JSONL, not CSV"
        ) if manifest_exists else f"missing {MANIFEST_PATH}",
    )
    if not manifest_exists:
        return finalize(results)

    raw_lines = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    blank_lines = [idx + 1 for idx, line in enumerate(raw_lines) if not line.strip()]
    add_result(
        results,
        "manifest_no_blank_lines",
        not blank_lines,
        "no blank lines" if not blank_lines else f"blank lines at {blank_lines[:10]}",
    )

    records = load_benchmark_manifest(MANIFEST_PATH)
    add_result(
        results,
        "manifest_not_empty",
        bool(records),
        f"records={len(records)}",
    )
    if not records:
        return finalize(results)

    missing_columns: dict[int, list[str]] = {}
    empty_required: dict[int, list[str]] = {}
    bad_image_size_rows: list[int] = []
    for idx, record in enumerate(records, start=1):
        missing = sorted(EXPECTED_COLUMNS - set(record))
        if missing:
            missing_columns[idx] = missing
        empties = [
            key
            for key in sorted(NON_EMPTY_COLUMNS)
            if not str(record.get(key, "")).strip()
        ]
        if empties:
            empty_required[idx] = empties
        image_size = record.get("image_size")
        if not (
            isinstance(image_size, list)
            and len(image_size) == 2
            and all(isinstance(v, int) and v > 0 for v in image_size)
        ):
            bad_image_size_rows.append(idx)

    add_result(
        results,
        "manifest_expected_columns",
        not missing_columns,
        "all expected columns present"
        if not missing_columns
        else f"first missing rows: {list(missing_columns.items())[:3]}",
    )
    add_result(
        results,
        "manifest_required_fields_non_empty",
        not empty_required,
        "required fields are populated"
        if not empty_required
        else f"first empty rows: {list(empty_required.items())[:3]}",
    )
    add_result(
        results,
        "manifest_image_sizes_valid",
        not bad_image_size_rows,
        "all image sizes valid"
        if not bad_image_size_rows
        else f"bad image_size rows: {bad_image_size_rows[:10]}",
    )

    unique_image_paths = sorted({str(record["image_path"]) for record in records})
    missing_images = [path for path in unique_image_paths if not Path(path).exists()]
    unreadable_images: list[str] = []
    for path in unique_image_paths[:]:
        if path in missing_images:
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except Exception:
            unreadable_images.append(path)
    add_result(
        results,
        "original_images_exist",
        not missing_images,
        f"existing={len(unique_image_paths)} missing={len(missing_images)}",
    )

    add_result(
        results,
        "manifest_matches_real_original_files",
        len(unique_image_paths) == len(records) and not missing_images,
        (
            f"manifest_records={len(records)} unique_original_files={len(unique_image_paths)}"
        ),
    )
    add_result(
        results,
        "original_images_readable",
        not unreadable_images,
        "all original images are readable"
        if not unreadable_images
        else f"unreadable image examples: {unreadable_images[:3]}",
    )

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    scenarios = load_scenarios(None)
    add_result(
        results,
        "summary_matches_manifest_count",
        int(summary.get("total_kept", -1)) == len(records),
        f"summary_total_kept={summary.get('total_kept')} manifest_records={len(records)}",
    )

    SCENARIO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    scenario_preview_failures: list[str] = []
    if records:
        smoke_record = records[0]
        for scenario in scenarios:
            try:
                payload = scenario_payload(smoke_record, scenario)
                out_path = SCENARIO_PREVIEW_DIR / f"{scenario.scenario_id}.png"
                payload["image"].save(out_path)
            except Exception as exc:  # pragma: no cover
                scenario_preview_failures.append(f"{scenario.scenario_id}:{exc}")
    add_result(
        results,
        "scenario_image_files_on_disk",
        not scenario_preview_failures
        and len(list(SCENARIO_PREVIEW_DIR.glob("*.png"))) == len(scenarios),
        (
            f"generated={len(list(SCENARIO_PREVIEW_DIR.glob('*.png')))} expected={len(scenarios)}; "
            "scenario previews were materialized by sanity_check because the benchmark stores only original files"
        ),
    )

    render_failures: list[str] = []
    render_sample = records[: min(10, len(records))]
    for scenario in scenarios:
        for record in render_sample:
            try:
                payload = scenario_payload(record, scenario)
                image = payload["image"]
                if not isinstance(image, Image.Image):
                    raise TypeError("payload image is not PIL.Image")
                if tuple(image.size) != tuple(record["image_size"]):
                    raise ValueError(
                        f"size mismatch {image.size} vs {tuple(record['image_size'])}"
                    )
            except Exception as exc:  # pragma: no cover
                render_failures.append(f"{scenario.scenario_id}:{record['example_id']}:{exc}")
                break
    add_result(
        results,
        "scenario_generation_works",
        not render_failures,
        "all sampled scenarios render successfully"
        if not render_failures
        else f"first failures: {render_failures[:3]}",
    )

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_rows: list[dict[str, Any]] = []
    candidate_visual_scenarios = [
        next(
            scenario
            for scenario in scenarios
            if scenario.ocr_strategy == "none"
            and scenario.image_strategy == image_strategy
            and scenario.context_window == 0
        )
        for image_strategy in ["black", "white", "blur"]
    ]
    for record in records:
        if len(preview_rows) >= min(5, len(records)):
            break
        original = Image.open(record["image_path"]).convert("RGB")
        chosen_payload = None
        differs = False
        for candidate_scenario in candidate_visual_scenarios:
            candidate_payload = scenario_payload(record, candidate_scenario)
            candidate_image = candidate_payload["image"]
            if crop_diff_exists(original, candidate_image, record["answer_bbox"]):
                chosen_payload = (candidate_scenario, candidate_image)
                differs = True
                break
        if chosen_payload is None:
            continue
        chosen_scenario, masked = chosen_payload
        example_id = str(record["example_id"])
        original_out = PREVIEW_DIR / f"{example_id}__original.png"
        masked_out = PREVIEW_DIR / f"{example_id}__masked.png"
        original.save(original_out)
        masked.save(masked_out)
        preview_rows.append(
            {
                "example_id": example_id,
                "image_size": list(original.size),
                "bbox": record["answer_bbox"],
                "scenario_id": chosen_scenario.scenario_id,
                "pixel_diff_in_bbox": differs,
                "original_preview": str(original_out.resolve()),
                "masked_preview": str(masked_out.resolve()),
            }
        )

    add_result(
        results,
        "visual_mask_check_sample_5",
        len(preview_rows) == min(5, len(records))
        and all(row["pixel_diff_in_bbox"] for row in preview_rows),
        json.dumps(preview_rows, ensure_ascii=False),
    )

    ocr_drop_scenario = next(
        scenario
        for scenario in scenarios
        if scenario.ocr_strategy == "drop"
        and scenario.image_strategy == "none"
        and scenario.context_window == 0
    )
    ocr_mask_scenario = next(
        scenario
        for scenario in scenarios
        if scenario.ocr_strategy == "mask"
        and scenario.image_strategy == "none"
        and scenario.context_window == 0
    )

    drop_failures: list[str] = []
    mask_failures: list[str] = []
    for record in records[: min(20, len(records))]:
        start_idx = int(record["answer_start_idx"])
        end_idx = int(record["answer_end_idx"])
        original_tokens = [str(token.get("text", "")) for token in record["ocr_tokens"]]

        drop_payload = scenario_payload(record, ocr_drop_scenario)
        drop_tokens = [str(token.get("text", "")) for token in drop_payload["ocr_tokens"]]
        expected_drop_tokens = original_tokens[:start_idx] + original_tokens[end_idx + 1 :]
        if drop_tokens != expected_drop_tokens:
            drop_failures.append(str(record["example_id"]))

        mask_payload = scenario_payload(record, ocr_mask_scenario)
        mask_tokens = [str(token.get("text", "")) for token in mask_payload["ocr_tokens"]]
        masked_span = mask_tokens[start_idx : end_idx + 1]
        if not masked_span or any(token != MASK_TOKEN for token in masked_span):
            mask_failures.append(str(record["example_id"]))

    add_result(
        results,
        "ocr_drop_removes_answer_span_sampled",
        not drop_failures,
        "sampled drop scenarios removed answer span tokens"
        if not drop_failures
        else f"drop failures: {drop_failures[:5]}",
    )
    add_result(
        results,
        "ocr_mask_replaces_with_redacted_sampled",
        not mask_failures,
        (
            f"sampled mask scenarios replaced answer span with {MASK_TOKEN}"
            if not mask_failures
            else f"mask failures: {mask_failures[:5]}"
        ),
    )

    return finalize(results)


def finalize(results: list[CheckResult]) -> int:
    print("Sanity Check Report")
    print(f"benchmark_dir={BENCHMARK_DIR}")
    print()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"  {result.details}")
    print()
    passed = sum(1 for result in results if result.passed)
    failed = sum(1 for result in results if not result.passed)
    print(f"Summary: passed={passed} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
