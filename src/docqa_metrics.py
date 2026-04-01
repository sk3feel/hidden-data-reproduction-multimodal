from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    overlap = sum((pred_counter & truth_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_metric_over_answers(
    prediction: str,
    answers: list[str],
) -> tuple[float, float]:
    normalized_answers = [str(answer) for answer in answers if str(answer).strip()]
    if not normalized_answers:
        normalized_answers = [""]
    em = max(exact_match_score(prediction, answer) for answer in normalized_answers)
    f1 = max(token_f1_score(prediction, answer) for answer in normalized_answers)
    return em, f1


def build_answer_pool(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        answer_groups[str(record["coarse_field_type"])].append(record)
    return dict(answer_groups)


def estimate_random_baseline(
    record: dict[str, Any],
    answer_pool: dict[str, list[dict[str, Any]]],
    num_samples: int = 20,
    seed: int = 42,
) -> tuple[float, float]:
    candidates = list(answer_pool.get(str(record["coarse_field_type"]), []))
    if not candidates:
        return 0.0, 0.0

    filtered = [
        candidate
        for candidate in candidates
        if str(candidate.get("example_id")) != str(record.get("example_id"))
    ]
    if filtered:
        candidates = filtered

    rng = random.Random(
        seed
        + int(record.get("local_row_id", 0))
        + sum(ord(ch) for ch in str(record.get("coarse_field_type", "")))
    )
    em_scores: list[float] = []
    f1_scores: list[float] = []
    for _ in range(max(num_samples, 1)):
        sampled = rng.choice(candidates)
        sampled_answer = str(sampled.get("answer", ""))
        em, f1 = best_metric_over_answers(
            prediction=sampled_answer,
            answers=list(record.get("answers") or [record.get("answer", "")]),
        )
        em_scores.append(em)
        f1_scores.append(f1)
    return sum(em_scores) / len(em_scores), sum(f1_scores) / len(f1_scores)
