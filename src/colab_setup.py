from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import login

from download_from_hf import download_dataset

DEFAULT_CLONE_DIR = Path("/content/course_work2026")
DEFAULT_REPO_URL = os.getenv(
    "COURSE_WORK2026_REPO_URL",
    "https://github.com/sk3feel/hidden-data-reproduction-multimodal.git",
)
DEFAULT_DATASET_REPO = os.getenv("HF_DATASET_REPO", "sk3feel/docvqa-privacy-data")
DEFAULT_COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "scfeel")


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_secret(name: str, default: str | None = None, required: bool = False) -> str | None:
    value: str | None = None

    if _in_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except Exception:
            value = None

    if not value:
        value = os.getenv(name, default)

    if required and not value:
        raise RuntimeError(f"Missing required secret: {name}")
    return value


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo_cloned(repo_url: str | None = None, clone_dir: Path = DEFAULT_CLONE_DIR) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / ".git").exists():
        return project_root

    if (clone_dir / ".git").exists():
        return clone_dir

    if not repo_url:
        raise RuntimeError("Repo is not cloned and COURSE_WORK2026_REPO_URL is empty.")

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", repo_url, str(clone_dir)])
    return clone_dir


def install_requirements(project_root: Path) -> None:
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements.txt: {requirements_path}")
    _run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


def login_hf_for_colab(hf_token: str) -> None:
    login(token=hf_token, add_to_git_credential=False)


def init_comet_experiment(
    api_key: str,
    workspace: str,
    project_name: str,
):
    import comet_ml

    experiment = comet_ml.Experiment(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
    )
    return experiment


def setup_colab(
    repo_url: str | None = None,
    clone_dir: Path = DEFAULT_CLONE_DIR,
    dataset_repo_id: str | None = None,
):
    repo_url = repo_url or get_secret("COURSE_WORK2026_REPO_URL", default=DEFAULT_REPO_URL)
    hf_token = get_secret("HF_TOKEN", required=True)
    dataset_repo_id = dataset_repo_id or get_secret(
        "HF_DATASET_REPO",
        default=DEFAULT_DATASET_REPO,
        required=True,
    )
    comet_api_key = get_secret("COMET_API_KEY", required=True)
    comet_workspace = get_secret(
        "COMET_WORKSPACE",
        default=DEFAULT_COMET_WORKSPACE,
        required=True,
    )
    comet_project_name = get_secret("COMET_PROJECT_NAME", default="qwen3-1", required=True)

    project_root = ensure_repo_cloned(repo_url=repo_url, clone_dir=clone_dir)
    install_requirements(project_root=project_root)
    login_hf_for_colab(hf_token=hf_token)

    download_summary = download_dataset(
        repo_id=dataset_repo_id,
        local_dir=project_root / "artifacts",
        token=hf_token,
    )
    experiment = init_comet_experiment(
        api_key=comet_api_key,
        workspace=comet_workspace,
        project_name=comet_project_name,
    )

    summary = {
        "project_root": str(project_root.resolve()),
        "dataset_repo_id": dataset_repo_id,
        "dataset_transport": "hf_hub_download + tar.gz bundle",
        "download_summary": download_summary,
        "comet_workspace": comet_workspace,
        "comet_project_name": comet_project_name,
        "comet_experiment_key": experiment.get_key() if hasattr(experiment, "get_key") else None,
        "comet_experiment_url": experiment.get_url() if hasattr(experiment, "get_url") else None,
    }
    return summary, experiment


def main() -> None:
    summary, _experiment = setup_colab()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
