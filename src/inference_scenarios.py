from __future__ import annotations

import re
from typing import Any

from PIL import Image

from docqa_benchmark import (
    _record_spans,
    redact_ocr_tokens_with_context,
    render_image_variant,
)

IMAGE_SCENARIOS = {
    "original": ("none", None),
    "img_none": ("none", None),
    "img_black": ("black", None),
    "img_white": ("white", None),
    "img_blur_10": ("blur", 10.0),
    "img_blur_20": ("blur", 20.0),
    "img_blur_50": ("blur", 50.0),
}
OCR_SCENARIOS = {
    "original": ("none", 0),
    "ocr_none": ("none", 0),
    "ocr_drop_k0": ("drop", 0),
    "ocr_drop_k20": ("drop", 20),
    "ocr_mask_k0": ("mask", 0),
    "ocr_mask_k20": ("mask", 20),
}


def _normalize_scenario_id(scenario_id: str) -> str:
    return re.sub(r"\s+", "", scenario_id.strip())


def _parse_legacy_combined_scenario(scenario_id: str) -> tuple[str, str, int]:
    if scenario_id == "original":
        return "original", "original", 0

    match = re.fullmatch(
        r"ocr_(none|drop|mask)__img_(none|black|white|blur(?:_\d+)?)__k_(\d+)",
        scenario_id,
    )
    if not match:
        raise ValueError(f"Unsupported scenario_id: {scenario_id}")

    ocr_strategy, image_part, context_window = match.groups()
    ocr_id = (
        f"ocr_{ocr_strategy}_k{context_window}"
        if ocr_strategy != "none"
        else "ocr_none"
    )
    image_id = "original" if image_part == "none" else f"img_{image_part}"
    return image_id, ocr_id, int(context_window)


def _split_scenario_id(scenario_id: str) -> tuple[str, str, int]:
    normalized = _normalize_scenario_id(scenario_id)
    if normalized in IMAGE_SCENARIOS or normalized in OCR_SCENARIOS:
        image_id = normalized if normalized in IMAGE_SCENARIOS else "original"
        ocr_id = normalized if normalized in OCR_SCENARIOS else "original"
        context_window = OCR_SCENARIOS.get(ocr_id, ("none", 0))[1]
        return image_id, ocr_id, context_window

    if "+" in normalized:
        parts = [part for part in normalized.split("+") if part]
        image_id = "original"
        ocr_id = "original"
        context_window = 0
        for part in parts:
            if part in IMAGE_SCENARIOS:
                image_id = part
            elif part in OCR_SCENARIOS:
                ocr_id = part
                context_window = OCR_SCENARIOS[ocr_id][1]
            else:
                raise ValueError(f"Unsupported scenario component: {part}")
        return image_id, ocr_id, context_window

    return _parse_legacy_combined_scenario(normalized)


def generate_scenario_image(record: dict[str, Any], scenario_id: str) -> Image.Image:
    image_id, _, context_window = _split_scenario_id(scenario_id)
    image_strategy, blur_sigma = IMAGE_SCENARIOS[image_id]
    return render_image_variant(
        image_path=record["image_path"],
        token_entries=record["ocr_tokens"],
        spans=_record_spans(record),
        image_strategy=image_strategy,
        context_window=context_window,
        blur_sigma=blur_sigma,
    )


def generate_scenario_ocr(record: dict[str, Any], scenario_id: str) -> str:
    _, ocr_id, context_window = _split_scenario_id(scenario_id)
    ocr_strategy, context_window = OCR_SCENARIOS[ocr_id]
    tokens = redact_ocr_tokens_with_context(
        token_entries=record["ocr_tokens"],
        spans=_record_spans(record),
        ocr_strategy=ocr_strategy,
        context_window=context_window,
    )
    return " ".join(
        str(token.get("text", "")).strip()
        for token in tokens
        if str(token.get("text", "")).strip()
    )
