from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docqa_benchmark import load_benchmark_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "finetuning_generative"
TRAIN_MANIFEST = PROJECT_ROOT / "artifacts" / "docqa_recovery" / "benchmark_train" / "manifest.jsonl"
VALIDATION_MANIFEST = PROJECT_ROOT / "artifacts" / "docqa_recovery" / "benchmark" / "manifest.jsonl"


def build_florence2_record(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record["question"]).strip()
    answer = str(record["answer"]).strip()
    return {
        "example_id": record["example_id"],
        "split": record["split"],
        "image_path": record["image_path"],
        "task_prompt": f"<DocVQA><Question>{question}</Question><Answer>",
        "answer": answer,
    }


def build_qwen2vl_record(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record["question"]).strip()
    answer = str(record["answer"]).strip()
    return {
        "example_id": record["example_id"],
        "split": record["split"],
        "image_path": record["image_path"],
        "chat_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record["image_path"]},
                    {
                        "type": "text",
                        "text": (
                            "Answer the document question with a short span copied from the "
                            f"document when possible.\nQuestion: {question}"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ],
        "answer": answer,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_manifest(
    manifest_path: Path,
    florence_output_path: Path,
    qwen_output_path: Path,
) -> dict[str, Any]:
    records = load_benchmark_manifest(manifest_path)
    florence_rows = [build_florence2_record(record) for record in records]
    qwen_rows = [build_qwen2vl_record(record) for record in records]

    _write_jsonl(florence_output_path, florence_rows)
    _write_jsonl(qwen_output_path, qwen_rows)

    return {
        "manifest_path": str(manifest_path.resolve()),
        "num_records": len(records),
        "florence_output_path": str(florence_output_path.resolve()),
        "qwen_output_path": str(qwen_output_path.resolve()),
    }


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    summaries = [
        export_manifest(
            manifest_path=TRAIN_MANIFEST,
            florence_output_path=output_dir / "train_florence2.jsonl",
            qwen_output_path=output_dir / "train_qwen2vl.jsonl",
        ),
        export_manifest(
            manifest_path=VALIDATION_MANIFEST,
            florence_output_path=output_dir / "validation_florence2.jsonl",
            qwen_output_path=output_dir / "validation_qwen2vl.jsonl",
        ),
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
