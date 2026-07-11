# Benchmark and Calibration

## Architecture

```
benchmarks/smoke/tasks.jsonl        16 cases - pipeline validation only
benchmarks/development/tasks.jsonl  48 cases - functional calibration testing
benchmarks/calibration/tasks.jsonl  32 cases - threshold search (split from development)
benchmarks/test/tasks.jsonl         16 cases - held-out evaluation (split from development)
```

The benchmark system evaluates routing quality by scoring model answers against
deterministic reference data. Calibration uses these scores to tune confidence
thresholds and model selection.

## Dataset Schema

Each JSONL line contains:

| Field | Required | Description |
|-------|----------|-------------|
| task_id | yes | Unique identifier |
| category | yes | One of 8 task categories |
| difficulty | yes | easy, medium, hard |
| prompt | yes | Input text |
| scoring_method | yes | Scorer to use |
| reference | varies | Expected answer |
| acceptable_answers | varies | List of valid answers |
| tolerance | no | Numeric tolerance |
| constraints | no | Summarization constraints |
| tests | varies | Code test cases |

## Scoring Methods

- `exact` - case-sensitive string match
- `normalized_exact` - casefold + whitespace normalization
- `acceptable_answers` - match any in list
- `numeric` - decimal comparison with tolerance
- `sentiment` - label extraction (positive/negative/neutral)
- `logic` - normalized match against reference/acceptable
- `ner` - precision/recall/F1 over entity pairs
- `summarization` - ROUGE-L F1 x constraint compliance
- `code_tests` - isolated subprocess execution

## Commands

### Validate dataset
```powershell
python -m benchmark validate-dataset --input benchmarks/development/tasks.jsonl
```

### Split dataset
```powershell
python -m benchmark split-dataset `
  --input benchmarks/development/tasks.jsonl `
  --calibration-output benchmarks/calibration/tasks.jsonl `
  --test-output benchmarks/test/tasks.jsonl `
  --calibration-ratio 0.7 --seed 42
```

### Run benchmark
```powershell
python -m benchmark run --dataset benchmarks/calibration/tasks.jsonl --strategy router --output artifacts/router_cal.jsonl
python -m benchmark run --dataset benchmarks/test/tasks.jsonl --strategy router --output artifacts/router_test.jsonl
```

### Run all allowed models
```powershell
python -m benchmark run-all-models `
  --dataset benchmarks/calibration/tasks.jsonl `
  --output-dir artifacts/model_runs `
  --allow-remote
```

### Calibrate
```powershell
python -m benchmark calibrate `
  --calibration-results artifacts/router_cal.jsonl `
  --test-results artifacts/router_test.jsonl `
  --output artifacts/routing_calibration.json `
  --minimum-accuracy 0.85 --objective tokens
```

### Compare strategies
```powershell
python -m benchmark compare `
  --results artifacts/router_cal.jsonl artifacts/local_cal.jsonl `
  --output artifacts/strategy_comparison.json
```

### Generate report
```powershell
python -m benchmark report --input artifacts/router_cal.jsonl --output artifacts/report.json
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| FIREWORKS_API_KEY | Required for remote calls |
| FIREWORKS_BASE_URL | API endpoint |
| ALLOWED_MODELS | Comma-separated exact model IDs |
| FIREWORKS_MODEL_CATALOG | JSON array of model pricing |
| MODEL_SELECTION_MODE | cost_first, balanced, quality_first |
| ALLOW_SYNTHETIC_CALIBRATION | Set true to load synthetic artifacts (default: false) |
| LOCAL_MODEL_ENABLED | Set false to skip local model |

## Synthetic Artifact Safety

All calibration artifacts include a `data_source` field:
- `"synthetic"` - generated from random data, rejected in production by default
- `"measured"` - generated from real model outputs, accepted in production

Synthetic artifacts are named `demo_*.synthetic.json` and are never loaded
by the production router unless `ALLOW_SYNTHETIC_CALIBRATION=true`.

## Production Calibration Loading

`ModelSelector` loads `artifacts/routing_calibration.json` at startup:
1. If file missing: static policy fallback
2. If file invalid JSON: static policy fallback
3. If `data_source: "synthetic"` and `ALLOW_SYNTHETIC_CALIBRATION != true`: static policy fallback
4. If valid measured data: calibrated cost-per-success selection

## Calibrated Model Selection Algorithm

1. Start with exact IDs from ALLOWED_MODELS
2. Filter models that cannot handle task/difficulty
3. Look up empirical profile (model+cat+diff then model+cat then model overall then static)
4. Compute `expected_cost_per_success = request_cost / max(accuracy, epsilon)`
5. Among models meeting minimum accuracy: pick lowest cost-per-success
6. Tie-break: higher accuracy, lower p95 latency, lower raw cost
7. If none meets accuracy: pick highest empirical accuracy
8. If no empirical data: fall back to static policy

## Limitations

- Development dataset has 48 cases - sufficient for functional testing, not
  statistically rigorous
- All current profiles are synthetic until real benchmarks are run
- Profiles with < 10 samples are flagged low_confidence
- No cross-validation or bootstrap confidence intervals
- Smoke dataset (16 cases) is for pipeline validation only, not calibration

## Real-Run Checklist

1. Set FIREWORKS_API_KEY, ALLOWED_MODELS, FIREWORKS_MODEL_CATALOG
2. Validate datasets
3. Run router on calibration split
4. Run router on test split
5. Run all models with run-all-models
6. Calibrate with calibration results
7. Compare strategies
8. Verify artifacts/routing_calibration.json has data_source: measured
