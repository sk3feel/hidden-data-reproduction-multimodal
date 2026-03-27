from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse, request

from datasets import Dataset, concatenate_datasets, load_dataset


ALLOWED_TYPES = {
    "DATE",
    "AMOUNT",
    "NAME",
    "ID",
    "ADDRESS",
    "PHONE",
    "ORG",
    "PERCENTAGE",
    "REFERENCE",
    "OTHER",
}


SYSTEM_PROMPT = """You classify document answer fields.
Task:
1) Choose one field type from this closed list:
DATE, AMOUNT, NAME, ID, ADDRESS, PHONE, ORG, PERCENTAGE, REFERENCE, OTHER
2) If no type fits, propose a new field type in snake_case.

Return strict JSON only:
{
  "field_type": "<TYPE_FROM_LIST_OR_NEW>",
  "new_type": "<snake_case_or_empty>",
  "reason": "<short reason>"
}
"""


@dataclass
class GigachatConfig:
    api_base: str
    auth_url: str
    model: str
    scope: str
    timeout_sec: float
    max_retries: int
    retry_delay_sec: float
    insecure_ssl: bool


def _build_ssl_context(insecure_ssl: bool) -> ssl.SSLContext | None:
    if not insecure_ssl:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
    ssl_context: ssl.SSLContext | None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_sec, context=ssl_context) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _http_post_form(
    url: str,
    form: dict[str, str],
    headers: dict[str, str],
    timeout_sec: float,
    ssl_context: ssl.SSLContext | None,
) -> dict[str, Any]:
    data = parse.urlencode(form).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_sec, context=ssl_context) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def get_access_token(
    credentials: str,
    cfg: GigachatConfig,
    ssl_context: ssl.SSLContext | None,
) -> str:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {credentials}",
    }
    payload = {"scope": cfg.scope}
    resp = _http_post_form(cfg.auth_url, payload, headers, cfg.timeout_sec, ssl_context)
    token = resp.get("access_token")
    if not token:
        raise RuntimeError(f"OAuth response has no access_token: {resp}")
    return str(token)


def build_user_prompt(question: str, answer: str) -> str:
    return (
        "Classify the answer field type for document QA.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Remember: prefer the closed list. "
        "Use a new snake_case type only if nothing from the list fits."
    )


def call_gigachat(
    access_token: str,
    cfg: GigachatConfig,
    question: str,
    answer: str,
    ssl_context: ssl.SSLContext | None,
) -> str:
    url = f"{cfg.api_base.rstrip('/')}/api/v1/chat/completions"
    payload = {
        "model": cfg.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, answer)},
        ],
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    resp = _http_post_json(url, payload, headers, cfg.timeout_sec, ssl_context)
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in completion response: {resp}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(f"No content in completion response: {resp}")
    return str(content).strip()


def parse_classification(raw_content: str) -> tuple[str, bool]:
    data: dict[str, Any] | None = None

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not data:
        token_match = re.search(
            r"\b(DATE|AMOUNT|NAME|ID|ADDRESS|PHONE|ORG|PERCENTAGE|REFERENCE|OTHER)\b",
            raw_content.upper(),
        )
        if token_match:
            return token_match.group(1), False
        return "OTHER", False

    raw_type = str(data.get("field_type", "")).strip()
    new_type = str(data.get("new_type", "")).strip()

    if raw_type in ALLOWED_TYPES:
        return raw_type, False

    if raw_type and _is_snake_case(raw_type):
        return raw_type, True

    if new_type and _is_snake_case(new_type):
        return new_type, True

    return "OTHER", False


def _is_snake_case(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{1,63}", value))


def select_answer(example: dict[str, Any]) -> str:
    answers = example.get("answers")
    if isinstance(answers, list):
        for item in answers:
            text = str(item).strip()
            if text:
                return text
    return ""


def _default_datasets_cache_dir() -> Path:
    custom = os.getenv("HF_DATASETS_CACHE", "").strip()
    if custom:
        return Path(custom)
    return Path.home() / ".cache" / "huggingface" / "datasets"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    # utf-8-sig removes BOM if file was created by tools that add it.
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_docvqa_from_local_cache(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    base = (
        _default_datasets_cache_dir()
        / "pixparse___docvqa-single-page-questions"
        / "default"
        / "0.0.0"
    )
    if not base.exists():
        raise FileNotFoundError(f"DocVQA cache not found at: {base}")

    hash_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not hash_dirs:
        raise FileNotFoundError(f"No hash directories in: {base}")

    # Use the freshest cache snapshot.
    hash_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = hash_dirs[0]
    shard_paths = sorted(chosen.glob(f"docvqa-single-page-questions-{split}-*.arrow"))
    if not shard_paths:
        raise FileNotFoundError(f"No .arrow shards for split={split!r} in: {chosen}")

    shard_datasets = [Dataset.from_file(str(path)) for path in shard_paths]
    ds = shard_datasets[0] if len(shard_datasets) == 1 else concatenate_datasets(shard_datasets)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return [dict(example) for example in ds]


def load_docvqa_examples(
    split: str,
    limit: int | None,
    allow_network: bool,
) -> list[dict[str, Any]]:
    if not allow_network:
        return load_docvqa_from_local_cache(split=split, limit=limit)

    ds = load_dataset("pixparse/docvqa-single-page-questions", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(example) for example in ds]


def classify_with_retries(
    question: str,
    answer: str,
    access_token: str,
    cfg: GigachatConfig,
    ssl_context: ssl.SSLContext | None,
) -> tuple[str, bool]:
    last_error: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = call_gigachat(access_token, cfg, question, answer, ssl_context)
            return parse_classification(raw)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_delay_sec * attempt)

    raise RuntimeError(f"Failed after retries: {last_error}") from last_error


def print_preview(rows: list[dict[str, Any]], n: int) -> None:
    print("\nFirst examples for manual check:")
    for idx, row in enumerate(rows[:n], start=1):
        print(
            f"[{idx:03d}] type={row['field_type']:<12} new={row['is_new_type']} "
            f"Q={row['question'][:90]!r} A={row['answer'][:70]!r}"
        )


def write_csv_with_fallback(rows: list[dict[str, Any]], output_csv: str) -> str:
    desired = Path(output_csv)
    fallback = Path.cwd() / desired.name

    for target in (desired, fallback):
        try:
            if target.parent:
                target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["question", "answer", "field_type", "is_new_type"],
                )
                writer.writeheader()
                writer.writerows(rows)
            if target != desired:
                print(f"[warn] Could not write to {desired}; used {target} instead.")
            return str(target)
        except OSError:
            continue

    raise RuntimeError(f"Failed to write CSV both to {desired} and {fallback}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label DocVQA answers by field type via GigaChat API."
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--output-csv",
        default="docvqa_field_types_pilot.csv",
    )
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--api-base", default="https://gigachat.devices.sberbank.ru")
    parser.add_argument(
        "--auth-url",
        default="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
    )
    parser.add_argument("--model", default="GigaChat")
    parser.add_argument("--scope", default=None)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=1.0)
    parser.add_argument("--credentials-env", default="GIGACHAT_CREDENTIALS")
    parser.add_argument("--access-token-env", default="GIGACHAT_ACCESS_TOKEN")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow datasets library to query Hugging Face Hub.",
    )
    args = parser.parse_args()

    # Load .env from project root so credentials can be kept out of notebooks/code.
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    if not args.allow_network:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    scope = args.scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    cfg = GigachatConfig(
        api_base=args.api_base,
        auth_url=args.auth_url,
        model=args.model,
        scope=scope,
        timeout_sec=args.timeout_sec,
        max_retries=max(args.max_retries, 1),
        retry_delay_sec=max(args.retry_delay_sec, 0.0),
        insecure_ssl=args.insecure_ssl,
    )

    examples = load_docvqa_examples(
        split=args.split,
        limit=args.limit,
        allow_network=args.allow_network,
    )
    if not examples:
        raise RuntimeError("No examples loaded from DocVQA.")

    ssl_context = _build_ssl_context(cfg.insecure_ssl)
    access_token = os.getenv(args.access_token_env, "").strip()

    if not args.dry_run and not access_token:
        credentials = os.getenv(args.credentials_env, "").strip()
        if not credentials:
            raise RuntimeError(
                f"Set either {args.access_token_env} or {args.credentials_env}."
            )
        access_token = get_access_token(credentials, cfg, ssl_context)

    rows: list[dict[str, Any]] = []
    errors = 0

    for idx, ex in enumerate(examples, start=1):
        question = str(ex.get("question", "")).strip()
        answer = select_answer(ex)

        if args.dry_run:
            field_type, is_new_type = "OTHER", False
        else:
            try:
                field_type, is_new_type = classify_with_retries(
                    question=question,
                    answer=answer,
                    access_token=access_token,
                    cfg=cfg,
                    ssl_context=ssl_context,
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                field_type, is_new_type = "OTHER", False
                print(f"[warn] #{idx} classification failed: {exc}")

        rows.append(
            {
                "question": question,
                "answer": answer,
                "field_type": field_type,
                "is_new_type": is_new_type,
            }
        )

        if idx % 10 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)}")
        time.sleep(args.sleep_sec)

    saved_csv = write_csv_with_fallback(rows, args.output_csv)

    counter = Counter(row["field_type"] for row in rows)
    print("\nField type stats:")
    for field_type, count in counter.most_common():
        print(f"{field_type}: {count}")

    print(f"\nSaved CSV: {saved_csv}")
    print(f"Errors: {errors}")

    print_preview(rows, n=max(args.preview, 0))


if __name__ == "__main__":
    main()
