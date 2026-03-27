from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPO_ID = os.getenv("HF_DATASET_REPO", "sk3feel/docvqa-privacy-data")
ARCHIVE_FILENAME = "artifacts_bundle.tar.gz"
EXTRACT_TARGETS = {
    "benchmark_train": Path("docqa_recovery/benchmark_train"),
    "benchmark": Path("docqa_recovery/benchmark"),
    "finetuning_generative": Path("finetuning_generative"),
}


def _safe_extract(archive_path: Path, extract_root: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe path inside archive: {member.name}")
        tar.extractall(path=extract_root)


def _replace_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def download_dataset(
    repo_id: str,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    token: str | None = None,
) -> dict[str, object]:
    local_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=ARCHIVE_FILENAME,
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir),
        )
    )
    extract_root = local_dir / "_hf_extract_tmp"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        _safe_extract(bundle_path, extract_root)

        extracted_paths: dict[str, str] = {}
        for archive_dir_name, relative_target in EXTRACT_TARGETS.items():
            source_dir = extract_root / archive_dir_name
            if not source_dir.exists():
                raise FileNotFoundError(f"Archive is missing expected directory: {archive_dir_name}")
            destination_dir = local_dir / relative_target
            _replace_tree(source_dir, destination_dir)
            extracted_paths[archive_dir_name] = str(destination_dir.resolve())
    finally:
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        if bundle_path.exists():
            bundle_path.unlink()

    return {
        "repo_id": repo_id,
        "local_dir": str(local_dir.resolve()),
        "archive_filename": ARCHIVE_FILENAME,
        "extracted_paths": extracted_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract project data artifacts from Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo, e.g. sk3feel/docvqa-privacy-data")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR), help="Where to place downloaded artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = download_dataset(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir).resolve(),
        token=os.getenv("HF_TOKEN"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
