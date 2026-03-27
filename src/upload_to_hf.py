from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, login

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_REPO_ID = os.getenv("HF_DATASET_REPO", "sk3feel/docvqa-privacy-data")
ARCHIVE_FILENAME = "artifacts_bundle.tar.gz"
ARCHIVE_ENTRIES = [
    ("docqa_recovery/benchmark_train", "benchmark_train"),
    ("docqa_recovery/benchmark", "benchmark"),
    ("finetuning_generative", "finetuning_generative"),
]


def _count_files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for entry in path.rglob("*") if entry.is_file())


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def build_archive(
    artifacts_root: Path,
    bundle_path: Path,
) -> list[dict[str, object]]:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    archived_entries: list[dict[str, object]] = []

    with tarfile.open(bundle_path, "w:gz") as tar:
        for relative_path, arcname in ARCHIVE_ENTRIES:
            source_path = artifacts_root / relative_path
            if not source_path.exists():
                raise FileNotFoundError(f"Required dataset entry is missing: {source_path}")

            tar.add(source_path, arcname=arcname)
            archived_entries.append(
                {
                    "source_path": relative_path,
                    "archive_path": arcname,
                    "files": _count_files(source_path),
                    "bytes": _size_bytes(source_path),
                }
            )

    return archived_entries


def upload_dataset(
    repo_id: str,
    hf_token: str,
    artifacts_root: Path = ARTIFACTS_ROOT,
    private: bool = True,
) -> dict[str, object]:
    login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    bundle_path = PROJECT_ROOT / ARCHIVE_FILENAME
    archived_entries = build_archive(artifacts_root=artifacts_root, bundle_path=bundle_path)

    try:
        archive_size_bytes = bundle_path.stat().st_size
        api.upload_file(
            path_or_fileobj=str(bundle_path),
            path_in_repo=ARCHIVE_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        if bundle_path.exists():
            bundle_path.unlink()

    return {
        "repo_id": repo_id,
        "artifacts_root": str(artifacts_root.resolve()),
        "private": private,
        "archive_filename": ARCHIVE_FILENAME,
        "archive_size_bytes": archive_size_bytes,
        "archived_entries": archived_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload project data artifacts to Hugging Face Hub as a single tar.gz dataset bundle.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target HF dataset repo, e.g. sk3feel/docvqa-privacy-data")
    parser.add_argument("--artifacts-root", default=str(ARTIFACTS_ROOT), help="Local artifacts root to archive from")
    parser.add_argument("--public", action="store_true", help="Create or update the dataset repo as public")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in the environment.")

    summary = upload_dataset(
        repo_id=args.repo_id,
        hf_token=hf_token,
        artifacts_root=Path(args.artifacts_root).resolve(),
        private=not args.public,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
