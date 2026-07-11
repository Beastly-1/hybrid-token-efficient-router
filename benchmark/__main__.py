"""CLI entry point: python -m benchmark [command] [options]

Commands:
  (default)          Aggregate routing JSONL logs into a telemetry summary.
  aggregate          Same as default.
  validate-dataset   Validate a benchmark dataset JSONL file.
  run                Run benchmark cases against an inference strategy.
  report             Generate accuracy report from benchmark results.
  calibrate          Calibrate routing thresholds from benchmark results.
  compare            Compare strategies from multiple result files.
  split-dataset      Split a dataset into calibration and test sets.
  run-all-models     Benchmark all ALLOWED_MODELS.
"""

import argparse
import json
import sys
from pathlib import Path

from benchmark.metrics import MetricsAggregator


def _cmd_aggregate(args) -> int:
    try:
        aggregator = MetricsAggregator(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    summary = aggregator.aggregate()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total = summary["requests"]["total"]
    fw = summary["fireworks"]["call_count"]
    print(f"Aggregated {total} records ({fw} Fireworks calls) -> {args.output}")
    return 0


def _cmd_validate_dataset(args) -> int:
    from benchmark.dataset import (
        DatasetValidationError,
        dataset_summary,
        load_dataset,
    )

    try:
        cases = load_dataset(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except DatasetValidationError as e:
        print(f"Validation failed:\n{e}", file=sys.stderr)
        return 1

    summary = dataset_summary(cases)
    print("Dataset valid")
    print(f"Total cases: {summary['total_cases']}")
    print("Categories:")
    for cat, count in summary["by_category"].items():
        print(f"  {cat}: {count}")
    print("Difficulties:")
    for diff, count in summary["by_difficulty"].items():
        print(f"  {diff}: {count}")
    print("Scoring methods:")
    for method, count in summary["by_scoring_method"].items():
        print(f"  {method}: {count}")
    return 0


def _cmd_run(args) -> int:
    from benchmark.dataset import DatasetValidationError, load_dataset
    from benchmark.runner import BenchmarkRunner

    try:
        cases = load_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except DatasetValidationError as e:
        print(f"Dataset validation failed:\n{e}", file=sys.stderr)
        return 1

    try:
        runner = BenchmarkRunner(
            strategy=args.strategy,
            allow_remote=args.allow_remote,
            run_id=args.run_id,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Running {len(cases)} cases with strategy '{args.strategy}' (run_id={runner.run_id})")
    results = runner.run_dataset(cases, args.output, max_cases=args.max_cases)

    scored = sum(1 for r in results if r.scoring_status == "scored")
    correct = sum(1 for r in results if r.correct is True)
    failed = sum(1 for r in results if not r.success)
    print(f"Done: {len(results)} cases, {scored} scored, {correct} correct, {failed} failures")
    print(f"Results -> {args.output}")
    return 0


def _cmd_report(args) -> int:
    from benchmark.report import generate_report, load_results

    try:
        records = load_results(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not records:
        print("Warning: no records found in input file.", file=sys.stderr)

    report = generate_report(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    overall = report.get("overall", {})
    print(f"Report: {overall.get('total_cases', 0)} cases, "
          f"accuracy={overall.get('accuracy')}, "
          f"mean_score={overall.get('mean_score')}")
    print(f"Report -> {args.output}")
    return 0


def _cmd_calibrate(args) -> int:
    from benchmark.calibration import calibrate, load_results

    try:
        cal_records = load_results(args.calibration_results)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    test_records = None
    if args.test_results:
        try:
            test_records = load_results(args.test_results)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    artifact = calibrate(
        calibration_records=cal_records,
        test_records=test_records,
        min_accuracy=args.minimum_accuracy,
        objective=args.objective,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    n_thresholds = len(artifact.get("thresholds", {}))
    n_profiles = len(artifact.get("model_profiles", []))
    print(f"Calibration complete: {n_thresholds} category thresholds, {n_profiles} model profiles")
    print(f"Artifact -> {args.output}")
    return 0


def _cmd_compare(args) -> int:
    from benchmark.strategy_comparison import compare_strategies

    result_files = {}
    for path in args.results:
        p = Path(path)
        name = p.stem
        result_files[name] = p

    try:
        comparison = compare_strategies(result_files)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    n = len(comparison.get("strategies", []))
    print(f"Compared {n} strategies -> {args.output}")
    return 0


def _cmd_split_dataset(args) -> int:
    """Split dataset into calibration and test sets with stratification."""
    import hashlib
    import random
    from benchmark.dataset import DatasetValidationError, load_dataset

    try:
        cases = load_dataset(args.input)
    except (FileNotFoundError, DatasetValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if len(cases) < 4:
        print("Error: dataset too small to split meaningfully.", file=sys.stderr)
        return 1

    # Stratify by category
    rng = random.Random(args.seed)
    by_cat: dict[str, list] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)

    cal_cases, test_cases = [], []
    for cat, cat_list in by_cat.items():
        shuffled = list(cat_list)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * args.calibration_ratio))
        cal_cases.extend(shuffled[:split_idx])
        test_cases.extend(shuffled[split_idx:])

    # Verify no overlap
    cal_ids = {c.task_id for c in cal_cases}
    test_ids = {c.task_id for c in test_cases}
    assert cal_ids.isdisjoint(test_ids)

    # Write outputs
    def _write_cases(cases_list, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for c in cases_list:
                record = {"task_id": c.task_id, "category": c.category,
                          "difficulty": c.difficulty, "prompt": c.prompt,
                          "scoring_method": c.scoring_method}
                if c.reference is not None:
                    record["reference"] = c.reference
                if c.acceptable_answers:
                    record["acceptable_answers"] = c.acceptable_answers
                if c.tolerance is not None:
                    record["tolerance"] = c.tolerance
                if c.constraints:
                    record["constraints"] = c.constraints
                if c.tests:
                    record["tests"] = c.tests
                if c.metadata:
                    record["metadata"] = c.metadata
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_cases(cal_cases, args.calibration_output)
    _write_cases(test_cases, args.test_output)

    # Fingerprint
    content = args.input.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()[:16]

    print(f"Split {len(cases)} cases (seed={args.seed}, ratio={args.calibration_ratio})")
    print(f"  Calibration: {len(cal_cases)} -> {args.calibration_output}")
    print(f"  Test: {len(test_cases)} -> {args.test_output}")
    print(f"  Dataset fingerprint: {fingerprint}")
    return 0


def _cmd_run_all_models(args) -> int:
    """Benchmark all ALLOWED_MODELS."""
    import os
    from config import ALLOWED_MODELS, FIREWORKS_API_KEY
    from benchmark.dataset import DatasetValidationError, load_dataset
    from benchmark.runner import BenchmarkRunner

    if not args.allow_remote:
        print("Error: --allow-remote required for run-all-models.", file=sys.stderr)
        return 1

    if not FIREWORKS_API_KEY:
        print("Error: FIREWORKS_API_KEY not set. Cannot make remote calls.", file=sys.stderr)
        return 1

    if not ALLOWED_MODELS:
        print("Error: ALLOWED_MODELS is empty. Set it to the model IDs to benchmark.", file=sys.stderr)
        return 1

    try:
        cases = load_dataset(args.dataset)
    except (FileNotFoundError, DatasetValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_models = len(ALLOWED_MODELS)
    print(f"Benchmarking {total_models} models on {len(cases)} cases")

    for i, model_id in enumerate(ALLOWED_MODELS, 1):
        # Safe filename from model ID
        safe_name = model_id.replace("/", "_").replace(" ", "_")
        output_path = args.output_dir / f"{safe_name}.jsonl"

        # Resume: skip if output already has expected count
        if output_path.exists():
            existing = sum(1 for line in output_path.open() if line.strip())
            expected = args.max_cases or len(cases)
            if existing >= expected:
                print(f"  [{i}/{total_models}] {model_id} - already complete ({existing} results)")
                continue

        print(f"  [{i}/{total_models}] {model_id} -> {output_path}")
        try:
            runner = BenchmarkRunner(
                strategy=f"remote:{model_id}",
                allow_remote=True,
                timeout_seconds=args.timeout_seconds,
            )
            results = runner.run_dataset(cases, output_path, max_cases=args.max_cases)
            correct = sum(1 for r in results if r.correct is True)
            print(f"    Done: {len(results)} cases, {correct} correct")
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    print(f"All model runs complete -> {args.output_dir}")
    return 0


def main() -> int:
    argv = sys.argv[1:]

    # Detect subcommand
    known_commands = {"aggregate", "validate-dataset", "run", "report", "calibrate", "compare",
                      "split-dataset", "run-all-models"}
    command = argv[0] if argv and argv[0] in known_commands else None

    if command == "validate-dataset":
        parser = argparse.ArgumentParser(description="Validate a benchmark dataset.")
        parser.add_argument("--input", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        return _cmd_validate_dataset(args)

    if command == "run":
        parser = argparse.ArgumentParser(description="Run benchmark cases.")
        parser.add_argument("--dataset", type=Path, required=True)
        parser.add_argument("--strategy", type=str, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--allow-remote", action="store_true", default=False)
        parser.add_argument("--max-cases", type=int, default=None)
        parser.add_argument("--timeout-seconds", type=float, default=120.0)
        parser.add_argument("--run-id", type=str, default=None)
        args = parser.parse_args(argv[1:])
        return _cmd_run(args)

    if command == "report":
        parser = argparse.ArgumentParser(description="Generate accuracy report.")
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        return _cmd_report(args)

    if command == "split-dataset":
        parser = argparse.ArgumentParser(description="Split dataset into calibration/test.")
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--calibration-output", type=Path, required=True)
        parser.add_argument("--test-output", type=Path, required=True)
        parser.add_argument("--calibration-ratio", type=float, default=0.7)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(argv[1:])
        return _cmd_split_dataset(args)

    if command == "run-all-models":
        parser = argparse.ArgumentParser(description="Benchmark all ALLOWED_MODELS.")
        parser.add_argument("--dataset", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--allow-remote", action="store_true", default=False)
        parser.add_argument("--max-cases", type=int, default=None)
        parser.add_argument("--timeout-seconds", type=float, default=120.0)
        args = parser.parse_args(argv[1:])
        return _cmd_run_all_models(args)

    if command == "calibrate":
        parser = argparse.ArgumentParser(description="Calibrate routing thresholds.")
        parser.add_argument("--calibration-results", type=Path, required=True)
        parser.add_argument("--test-results", type=Path, default=None)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--minimum-accuracy", type=float, default=0.90)
        parser.add_argument("--objective", type=str, default="tokens",
                            choices=["tokens", "cost", "latency"])
        args = parser.parse_args(argv[1:])
        return _cmd_calibrate(args)

    if command == "compare":
        parser = argparse.ArgumentParser(description="Compare strategies.")
        parser.add_argument("--results", type=Path, nargs="+", required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        return _cmd_compare(args)

    if command == "aggregate":
        argv = argv[1:]

    # Default: aggregate (backwards compatible with --input --output)
    parser = argparse.ArgumentParser(
        description="Aggregate routing JSONL logs into a telemetry summary."
    )
    parser.add_argument("--input", type=Path, default=Path("logs/routing.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/telemetry_summary.json"))
    args = parser.parse_args(argv)
    return _cmd_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
