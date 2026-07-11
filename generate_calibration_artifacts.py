"""Offline demonstration: generate synthetic results and all final artifacts.

Usage: python generate_calibration_artifacts.py

No Fireworks API calls are made. All data is synthetic.
"""

import json
import random
import time
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODELS = [
    "accounts/fireworks/models/gemma-4-31b-it",
    "accounts/fireworks/models/gemma-4-26b-a4b-it",
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
]

CATEGORIES = ["math", "factual", "sentiment", "summarization", "ner", "code_debug", "logic", "code_generation"]
DIFFICULTIES = ["easy", "medium", "hard"]

random.seed(42)


def _synthetic_record(category, difficulty, strategy, model=None, route_source="fireworks"):
    base_acc = {"easy": 0.9, "medium": 0.75, "hard": 0.55}[difficulty]
    model_bonus = {
        MODELS[0]: 0.05, MODELS[1]: 0.03, MODELS[2]: 0.08,
        MODELS[3]: 0.06, MODELS[4]: 0.04,
    }.get(model, 0.0)
    cat_bonus = {"math": 0.0, "code_generation": -0.05, "factual": 0.05,
                 "sentiment": 0.1, "ner": -0.03, "summarization": 0.02,
                 "logic": -0.02, "code_debug": -0.04}.get(category, 0.0)

    acc_prob = min(0.99, max(0.1, base_acc + model_bonus + cat_bonus))
    correct = random.random() < acc_prob
    score = 1.0 if correct else random.uniform(0.0, 0.5)
    confidence = random.uniform(0.5, 0.95)
    tokens = random.randint(80, 400) if route_source == "fireworks" else 0
    cost = tokens * 0.2 / 1_000_000 if route_source == "fireworks" else 0.0
    latency = random.uniform(20, 200) if route_source == "fireworks" else random.uniform(5, 50)

    return {
        "task_id": f"{category}_{difficulty}_{random.randint(1000, 9999)}",
        "category": category, "difficulty": difficulty,
        "strategy": strategy, "scoring_method": "exact",
        "scoring_status": "scored", "correct": correct, "score": score,
        "route_score": confidence, "route_source": route_source,
        "selected_model": model,
        "total_tokens": tokens,
        "prompt_tokens": int(tokens * 0.7) if tokens else None,
        "completion_tokens": int(tokens * 0.3) if tokens else None,
        "actual_remote_cost": cost if cost > 0 else None,
        "estimated_remote_cost": cost * 1.1 if cost > 0 else None,
        "total_latency_ms": latency,
        "remote_latency_ms": latency * 0.8 if route_source == "fireworks" else None,
        "success": True,
    }


def generate_strategy_results(strategy_name, n_per_cat=15):
    records = []
    for cat in CATEGORIES:
        for diff in DIFFICULTIES:
            for _ in range(n_per_cat):
                if strategy_name == "always_local":
                    r = _synthetic_record(cat, diff, strategy_name, route_source="local")
                elif strategy_name == "always_remote":
                    r = _synthetic_record(cat, diff, strategy_name, model=random.choice(MODELS), route_source="fireworks")
                elif strategy_name == "current_router":
                    if random.random() < 0.4:
                        r = _synthetic_record(cat, diff, strategy_name, route_source="local")
                    else:
                        r = _synthetic_record(cat, diff, strategy_name, model=random.choice(MODELS), route_source="fireworks")
                elif strategy_name == "cost_first":
                    r = _synthetic_record(cat, diff, strategy_name, model=MODELS[0], route_source="fireworks")
                elif strategy_name == "accuracy_first":
                    r = _synthetic_record(cat, diff, strategy_name, model=MODELS[2], route_source="fireworks")
                else:
                    if random.random() < 0.3:
                        r = _synthetic_record(cat, diff, strategy_name, route_source="local")
                    else:
                        r = _synthetic_record(cat, diff, strategy_name, model=random.choice(MODELS[:3]), route_source="fireworks")
                records.append(r)
    return records


def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    print("=" * 60)
    print("OFFLINE CALIBRATION DEMONSTRATION")
    print("=" * 60)

    strategies = ["always_local", "always_remote", "current_router",
                  "cost_first", "accuracy_first", "balanced", "calibrated"]
    result_files = {}
    for strat in strategies:
        records = generate_strategy_results(strat)
        path = ARTIFACTS_DIR / f"{strat}_results.jsonl"
        write_jsonl(records, path)
        result_files[strat] = path
        print(f"  Generated {len(records)} records for '{strat}' -> {path}")

    # Calibration
    print("\n--- CALIBRATION ---")
    from benchmark.calibration import calibrate, load_results
    cal_records = load_results(ARTIFACTS_DIR / "current_router_results.jsonl")
    test_records = load_results(ARTIFACTS_DIR / "balanced_results.jsonl")
    calibration_artifact = calibrate(cal_records, test_records, min_accuracy=0.80, objective="tokens", data_source="synthetic")
    cal_path = ARTIFACTS_DIR / "demo_routing_calibration.synthetic.json"
    cal_path.write_text(json.dumps(calibration_artifact, indent=2) + "\n", encoding="utf-8")
    print(f"  {len(calibration_artifact['thresholds'])} thresholds, {len(calibration_artifact['model_profiles'])} profiles -> {cal_path}")

    # Strategy comparison
    print("\n--- STRATEGY COMPARISON ---")
    from benchmark.strategy_comparison import compare_strategies
    comparison = compare_strategies(result_files)
    cmp_path = ARTIFACTS_DIR / "demo_strategy_comparison.synthetic.json"
    cmp_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"  Pareto (tokens): {comparison['pareto_efficient']['accuracy_vs_tokens']}")
    for m in comparison["strategies"]:
        print(f"    {m['strategy']:20s}: acc={m['accuracy']:.3f}, tokens={m['fireworks_total_tokens']:>6d}, cost=${m['actual_cost']:.4f}")

    # Final report JSON
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration": calibration_artifact,
        "strategy_comparison": comparison,
    }
    (ARTIFACTS_DIR / "demo_final_benchmark_report.synthetic.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Final report Markdown
    md = ["# Final Benchmark & Calibration Report (SYNTHETIC DEMO)", "",
          f"Generated: {report['generated_at']}", "",
          "## Thresholds", "",
          "| Category | Threshold | Accuracy | Met |",
          "|----------|-----------|----------|-----|"]
    for cat, e in sorted(calibration_artifact["thresholds"].items()):
        md.append(f"| {cat} | {e['threshold']:.2f} | {e['accuracy']:.3f} | {'✓' if e['constraint_met'] else '✗'} |")
    md.extend(["", "## Strategies", "",
               "| Strategy | Accuracy | Tokens | Cost |",
               "|----------|----------|--------|------|"])
    for m in sorted(comparison["strategies"], key=lambda x: -(x["accuracy"] or 0)):
        md.append(f"| {m['strategy']} | {m['accuracy']:.3f} | {m['fireworks_total_tokens']} | ${m['actual_cost']:.4f} |")
    md.extend(["", "## Limitations", "",
               "- Synthetic data only; re-run with real models for production calibration.",
               "- Small sample sizes; statistical significance not claimed.",
               "- Profiles with < 10 samples are marked low-confidence.", ""])
    (ARTIFACTS_DIR / "demo_final_benchmark_report.synthetic.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n  All artifacts in {ARTIFACTS_DIR}/")
    print("  No Fireworks API calls were made.")


if __name__ == "__main__":
    main()
