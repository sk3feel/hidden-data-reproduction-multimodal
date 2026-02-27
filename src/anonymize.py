from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    from .load_data import extract_ocr_tokens
except ImportError:
    from load_data import extract_ocr_tokens  # type: ignore


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_text(token: Any) -> str:
    if isinstance(token, dict):
        return str(token.get("text", ""))
    return str(token)


def find_answer_span(ocr_tokens: list[Any], answer: str) -> tuple[int | None, int | None, bool]:
    """
    Ищет точное совпадение нормализованного ответа в OCR-токенах.
    Возвращает (start_idx, end_idx, found), где end_idx включителен.
    """
    normalized_answer = normalize_text(answer)
    answer_parts = normalized_answer.split()
    if not answer_parts:
        return None, None, False

    # Нормализуем OCR и разворачиваем токены на под-токены.
    # Это закрывает случаи вида:
    # - ответ "3,700" -> ["3", "700"], OCR токен "3,700"
    # - OCR разбиение "$117", ",", "047"
    flat_tokens: list[str] = []
    flat_to_orig_idx: list[int] = []
    for orig_idx, token in enumerate(ocr_tokens):
        parts = normalize_text(_token_text(token)).split()
        for part in parts:
            flat_tokens.append(part)
            flat_to_orig_idx.append(orig_idx)

    n = len(answer_parts)
    for start in range(len(flat_tokens) - n + 1):
        window = flat_tokens[start : start + n]
        if window == answer_parts:
            start_orig = flat_to_orig_idx[start]
            end_orig = flat_to_orig_idx[start + n - 1]
            return start_orig, end_orig, True

    return None, None, False


def _bbox_to_xyxy(bbox: Any, image_size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    # Случай 1: [x1, y1, x2, y2]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(
        isinstance(v, (int, float)) for v in bbox
    ):
        x1, y1, x2, y2 = bbox
        coords = [float(x1), float(y1), float(x2), float(y2)]
    # Случай 2: плоский полигон [x1, y1, x2, y2, ...]
    elif (
        isinstance(bbox, (list, tuple))
        and len(bbox) >= 6
        and len(bbox) % 2 == 0
        and all(isinstance(v, (int, float)) for v in bbox)
    ):
        xs = [float(v) for v in bbox[0::2]]
        ys = [float(v) for v in bbox[1::2]]
        coords = [min(xs), min(ys), max(xs), max(ys)]
    # Случай 3: полигон [[x,y], [x,y], ...]
    elif (
        isinstance(bbox, (list, tuple))
        and len(bbox) >= 2
        and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in bbox)
    ):
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        coords = [min(xs), min(ys), max(xs), max(ys)]
    else:
        raise ValueError(f"Unsupported bbox format: {bbox}")

    if image_size is not None:
        width, height = image_size
        # Масштабируем только нормализованные координаты [0, 1].
        # В DocVQA bbox обычно уже в пикселях.
        if max(coords) <= 1.0:
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]

    x1, y1, x2, y2 = [int(round(v)) for v in coords]
    return x1, y1, x2, y2


def span_bbox_from_tokens(token_entries: list[dict[str, Any]], start_idx: int, end_idx: int) -> tuple[int, int, int, int] | None:
    bboxes = []
    for token in token_entries[start_idx : end_idx + 1]:
        bbox = token.get("bbox")
        if bbox is None:
            continue
        try:
            bboxes.append(_bbox_to_xyxy(bbox))
        except ValueError:
            continue

    if not bboxes:
        return None

    xs1, ys1, xs2, ys2 = zip(*bboxes)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def evaluate_match_rate(dataset_subset: list[dict[str, Any]], output_path: str = "outputs/stats/answer_match_rate.json") -> dict[str, Any]:
    total = 0
    matched = 0
    details: list[dict[str, Any]] = []

    for example in dataset_subset:
        answers = example.get("answers") or []
        if not answers:
            continue

        token_entries = extract_ocr_tokens(example)
        ocr_texts = [t["text"] for t in token_entries]

        answer = str(answers[0])
        start_idx, end_idx, found = find_answer_span(ocr_texts, answer)

        total += 1
        matched += int(found)
        details.append(
            {
                "question": example.get("question"),
                "answer": answer,
                "found": found,
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
        )

    result = {
        "total_checked": total,
        "matched": matched,
        "match_rate": (matched / total) if total else 0.0,
        "note": "Точное совпадение нормализованной n-граммы по OCR-токенам (используется первый вариант ответа).",
        "examples": details[:20],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def mask_image(image: Image.Image, bbox: Any) -> Image.Image:
    """
    Закрашивает прямоугольник bbox черным и возвращает копию изображения.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = _bbox_to_xyxy(bbox, image_size=out.size)
    draw.rectangle([x1, y1, x2, y2], fill="black")
    return out


def redact_ocr_tokens(ocr_tokens: list[Any], start_idx: int, end_idx: int, strategy: str = "drop") -> list[Any]:
    if strategy == "drop":
        return list(ocr_tokens[:start_idx]) + list(ocr_tokens[end_idx + 1 :])

    # strategy == "mask"
    out: list[Any] = []
    for i, token in enumerate(ocr_tokens):
        if start_idx <= i <= end_idx:
            if isinstance(token, dict):
                t = dict(token)
                t["text"] = "[REDACTED]"
                out.append(t)
            else:
                out.append("[REDACTED]")
        else:
            out.append(token)
    return out


