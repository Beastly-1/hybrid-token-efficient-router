"""Threshold calibration from benchmark results.

Searches candidate local-confidence thresholds per task category to minimize
a selected objective (tokens, cost, latency) while meeting a minimum accuracy.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Minimum samples required for difficulty-specific thresholds
_MIN_DIFFICULTY_SAMPLES = 10
# Minimum samples for a category to be calibrated at all
_MIN_CATEGORY_SAMPLES = 5
# Confidence floor to avoid division by zero
_EPSILON = 1e-6
# Low-confidence sample threshold for model profiles
_LOW_CONFIDENCE_THRESHOLD = 10


@dataclass
class ThresholdCandidate:
    threshold: float
    accuracy: float
    local_acceptance_rate: float
    incorrect_local_rate: float
    fireworks_call_rate: float
    total_fireworks_tokens: int
    total_cost: float
    average_latency_ms: float
    p95_latency_ms: float


@dataclass
class CalibrationResult:
    category: str
    selected_threshold: float
    constraint_met: bool
    objective_value: float
    candidate: ThresholdCandidate
    difficulty_thresholds: dict = field(default_factory=dict)
    sample_count: int = 0


@dataclass
class ModelProfile:
    model_id: str
    category: str
    difficulty: str
    sample_count: int
    accuracy: float
    mean_score: float
    average_cost: float
    average_tokens: float
    average_latency_ms: float
    p95_latency_ms: float
    invalid_output_rate: float
    cost_per_correct: float
    low_confidence: bool


def load_results(path: Path) -> list[dict]:
    """Load benchmark result JSONL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    records.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return records


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(math.ceil(p / 100 * len(s))) - 1))
    return s[k]


def _evaluate_threshold(records: list[dict], threshold: float) -> ThresholdCandidate:
    """Evaluate a single threshold against a set of records."""
    total = len(records)
    if total == 0:
        return ThresholdCandidate(threshold, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    local_accepted = 0
    local_correct = 0
    local_incorrect = 0
    fireworks_calls = 0
    fireworks_tokens = 0
    total_cost = 0.0
    correct_count = 0
    latencies = []

    for r in records:
        confidence = r.get("route_score") or 0.0
        is_correct = r.get("correct", False)
        tokens = r.get("total_tokens") or 0
        cost = r.get("actual_remote_cost") or r.get("estimated_remote_cost") or 0.0
        latency = r.get("total_latency_ms") or 0.0

        if confidence >= threshold:
            # Local accepts
            local_accepted += 1
            if is_correct:
                local_correct += 1
                correct_count += 1
            else:
                local_incorrect += 1
            latencies.append(latency)
        else:
            # Falls through to Fireworks
            fireworks_calls += 1
            fireworks_tokens += tokens
            total_cost += cost
            if is_correct:
                correct_count += 1
            latencies.append(latency)

    accuracy = correct_count / total if total > 0 else 0.0
    local_rate = local_accepted / total if total > 0 else 0.0
    incorrect_local = local_incorrect / total if total > 0 else 0.0
    fw_rate = fireworks_calls / total if total > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = _percentile(latencies, 95)

    return ThresholdCandidate(
        threshold=threshold,
        accuracy=accuracy,
        local_acceptance_rate=local_rate,
        incorrect_local_rate=incorrect_local,
        fireworks_call_rate=fw_rate,
        total_fireworks_tokens=fireworks_tokens,
        total_cost=total_cost,
        average_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
    )


def _objective_value(candidate: ThresholdCandidate, objective: str) -> float:
    """Extract the objective metric from a candidate."""
    if objective == "tokens":
        return candidate.total_fireworks_tokens
    elif objective == "cost":
        return candidate.total_cost
    elif objective == "latency":
        return candidate.average_latency_ms
    return candidate.total_fireworks_tokens


def calibrate_category(
    records: list[dict],
    category: str,
    min_accuracy: float = 0.90,
    objective: str = "tokens",
    step: float = 0.05,
) -> Optional[CalibrationResult]:
    """Search thresholds for a single category."""
    cat_records = [r for r in records if r.get("category") == category]
    if len(cat_records) < _MIN_CATEGORY_SAMPLES:
        return None

    # Generate candidates
    thresholds = [round(t * step, 4) for t in range(int(1.0 / step) + 1)]
    candidates = [_evaluate_threshold(cat_records, t) for t in thresholds]

    # Find best meeting accuracy constraint
    valid = [c for c in candidates if c.accuracy >= min_accuracy]
    if valid:
        best = min(valid, key=lambda c: (_objective_value(c, objective), -c.accuracy))
        constraint_met = True
    else:
        # Fallback: highest accuracy, objective as tie-breaker
        best = min(candidates, key=lambda c: (-c.accuracy, _objective_value(c, objective)))
        constraint_met = False

    # Difficulty-specific thresholds
    difficulty_thresholds = {}
    for diff in ("easy", "medium", "hard"):
        diff_records = [r for r in cat_records if r.get("difficulty") == diff]
        if len(diff_records) >= _MIN_DIFFICULTY_SAMPLES:
            diff_candidates = [_evaluate_threshold(diff_records, t) for t in thresholds]
            diff_valid = [c for c in diff_candidates if c.accuracy >= min_accuracy]
            if diff_valid:
                diff_best = min(diff_valid, key=lambda c: (_objective_value(c, objective), -c.accuracy))
            else:
                diff_best = min(diff_candidates, key=lambda c: (-c.accuracy, _objective_value(c, objective)))
            difficulty_thresholds[diff] = diff_best.threshold

    return CalibrationResult(
        category=category,
        selected_threshold=best.threshold,
        constraint_met=constraint_met,
        objective_value=_objective_value(best, objective),
        candidate=best,
        difficulty_thresholds=difficulty_thresholds,
        sample_count=len(cat_records),
    )


def build_model_profiles(records: list[dict]) -> list[ModelProfile]:
    """Build empirical accuracy profiles per model/category/difficulty."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        model = r.get("selected_model")
        if not model:
            continue
        cat = r.get("category", "unknown")
        diff = r.get("difficulty", "unknown")
        groups.setdefault((model, cat, diff), []).append(r)

    profiles = []
    for (model, cat, diff), recs in groups.items():
        scored = [r for r in recs if r.get("scoring_status") == "scored"]
        correct = [r for r in scored if r.get("correct") is True]
        scores = [r["score"] for r in scored if r.get("score") is not None]
        costs = [r.get("actual_remote_cost") or r.get("estimated_remote_cost") or 0.0 for r in recs]
        tokens = [r.get("total_tokens") or 0 for r in recs]
        latencies = [r.get("total_latency_ms") or 0.0 for r in recs]
        invalid = [r for r in recs if r.get("scoring_status") == "invalid_answer"]

        n = len(recs)
        acc = len(correct) / len(scored) if scored else 0.0
        mean_sc = sum(scores) / len(scores) if scores else 0.0
        avg_cost = sum(costs) / n if n else 0.0
        avg_tok = sum(tokens) / n if n else 0.0
        avg_lat = sum(latencies) / n if n else 0.0
        p95_lat = _percentile(latencies, 95)
        inv_rate = len(invalid) / n if n else 0.0
        cpc = avg_cost / max(acc, _EPSILON) if acc > 0 else avg_cost / _EPSILON

        profiles.append(ModelProfile(
            model_id=model, category=cat, difficulty=diff,
            sample_count=n, accuracy=acc, mean_score=mean_sc,
            average_cost=avg_cost, average_tokens=avg_tok,
            average_latency_ms=avg_lat, p95_latency_ms=p95_lat,
            invalid_output_rate=inv_rate, cost_per_correct=cpc,
            low_confidence=n < _LOW_CONFIDENCE_THRESHOLD,
        ))
    return profiles


def calibrate(
    calibration_records: list[dict],
    test_records: Optional[list[dict]] = None,
    min_accuracy: float = 0.90,
    objective: str = "tokens",
    data_source: Optional[str] = None,
) -> dict:
    """Run full calibration and return artifact dict.

    calibration_records: used to search thresholds
    test_records: used to evaluate (prevents leakage); if None, skipped
    data_source: 'synthetic' or 'measured' (default 'measured')
    """
    categories = sorted(set(r.get("category", "unknown") for r in calibration_records))

    results = {}
    for cat in categories:
        result = calibrate_category(calibration_records, cat, min_accuracy, objective)
        if result:
            results[cat] = result

    # Build model profiles from calibration data
    profiles = build_model_profiles(calibration_records)

    # Evaluate on test set if provided
    test_evaluation = None
    if test_records:
        test_evaluation = _evaluate_on_test(test_records, results)

    # Build artifact
    import time
    artifact = {
        "schema_version": 2,
        "data_source": data_source or "measured",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "minimum_accuracy": min_accuracy,
        "objective": objective,
        "thresholds": {},
        "model_profiles": [],
        "strategy_summary": {},
    }

    for cat, res in results.items():
        entry = {
            "threshold": res.selected_threshold,
            "constraint_met": res.constraint_met,
            "sample_count": res.sample_count,
            "accuracy": res.candidate.accuracy,
            "local_acceptance_rate": res.candidate.local_acceptance_rate,
            "incorrect_local_rate": res.candidate.incorrect_local_rate,
            "fireworks_call_rate": res.candidate.fireworks_call_rate,
            "total_fireworks_tokens": res.candidate.total_fireworks_tokens,
            "total_cost": res.candidate.total_cost,
            "objective_value": res.objective_value,
        }
        if res.difficulty_thresholds:
            entry["difficulty_thresholds"] = res.difficulty_thresholds
        artifact["thresholds"][cat] = entry

    for p in profiles:
        artifact["model_profiles"].append({
            "model_id": p.model_id,
            "category": p.category,
            "difficulty": p.difficulty,
            "sample_count": p.sample_count,
            "accuracy": p.accuracy,
            "mean_score": p.mean_score,
            "average_cost": p.average_cost,
            "average_tokens": p.average_tokens,
            "average_latency_ms": p.average_latency_ms,
            "p95_latency_ms": p.p95_latency_ms,
            "invalid_output_rate": p.invalid_output_rate,
            "cost_per_correct": p.cost_per_correct,
            "low_confidence": p.low_confidence,
        })

    if test_evaluation:
        artifact["test_evaluation"] = test_evaluation

    return artifact


def _evaluate_on_test(test_records: list[dict], calibration_results: dict) -> dict:
    """Evaluate calibrated thresholds on a separate test set."""
    evaluation = {}
    categories = sorted(set(r.get("category", "unknown") for r in test_records))
    for cat in categories:
        cat_records = [r for r in test_records if r.get("category") == cat]
        if not cat_records:
            continue
        if cat in calibration_results:
            threshold = calibration_results[cat].selected_threshold
        else:
            threshold = 0.85  # default
        candidate = _evaluate_threshold(cat_records, threshold)
        evaluation[cat] = {
            "threshold_used": threshold,
            "sample_count": len(cat_records),
            "accuracy": candidate.accuracy,
            "local_acceptance_rate": candidate.local_acceptance_rate,
            "fireworks_call_rate": candidate.fireworks_call_rate,
        }
    return evaluation
