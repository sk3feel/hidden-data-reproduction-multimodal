from __future__ import annotations

from io import BytesIO
from typing import Any

from datasets import load_dataset
from PIL import Image


DEFAULT_DATASET_NAME = "pixparse/docvqa-single-page-questions"


def _to_pil_image(image_obj: Any) -> Image.Image | None:
    if image_obj is None:
        return None

    if isinstance(image_obj, Image.Image):
        return image_obj

    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert("RGB")

    # Фолбэк для array-like объектов.
    return Image.fromarray(image_obj).convert("RGB")


def extract_ocr_tokens(example: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Разворачивает OCR-слова в порядок чтения.

    Возвращает список вида {"text": str, "bbox": Any}.
    Для pixparse/docvqa-single-page-questions OCR хранится в:
    example["ocr_results"]["lines"][...]["words"][...]
    """
    tokens: list[dict[str, Any]] = []

    ocr_results = example.get("ocr_results") or {}
    lines = ocr_results.get("lines") or []

    for line in lines:
        for word in line.get("words") or []:
            text = str(word.get("text", "")).strip()
            if not text:
                continue
            bbox = word.get("bounding_box")
            if bbox is None:
                bbox = word.get("bbox")
            tokens.append({"text": text, "bbox": bbox})

    return tokens


def load_docvqa(split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    """
    Загружает DocVQA из Hugging Face и возвращает обычный список словарей.

    По умолчанию используется `pixparse/docvqa-single-page-questions`.
    """
    ds = load_dataset(DEFAULT_DATASET_NAME, split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return [dict(example) for example in ds]


def show_example(example: dict[str, Any]) -> None:
    """
    Печатает вопрос/ответы и первые OCR-токены, затем показывает изображение.
    """
    import matplotlib.pyplot as plt

    image = _to_pil_image(example.get("image"))
    question = example.get("question")
    answers = example.get("answers")
    tokens = extract_ocr_tokens(example)

    print("Вопрос:", question)
    print("Ответы:", answers)
    print("OCR токены (первые 15):")
    for i, token in enumerate(tokens[:15]):
        print(f"  {i:02d}: {token['text']} | bbox={token.get('bbox')}")

    if image is None:
        print("Изображение недоступно")
        return

    plt.figure(figsize=(8, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title(str(question))
    plt.show()
