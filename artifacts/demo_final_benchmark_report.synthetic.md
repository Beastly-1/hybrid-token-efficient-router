# Final Benchmark & Calibration Report

Generated: 2026-07-11T19:53:47+0530

## Thresholds

| Category | Threshold | Accuracy | Met |
|----------|-----------|----------|-----|
| code_debug | 0.00 | 0.800 | ✓ |
| code_generation | 0.00 | 0.689 | ✗ |
| factual | 0.00 | 0.844 | ✓ |
| logic | 0.00 | 0.711 | ✗ |
| math | 0.00 | 0.756 | ✗ |
| ner | 0.00 | 0.733 | ✗ |
| sentiment | 0.00 | 0.867 | ✓ |
| summarization | 0.00 | 0.844 | ✓ |

## Strategies

| Strategy | Accuracy | Tokens | Cost |
|----------|----------|--------|------|
| cost_first | 0.817 | 85721 | $0.0171 |
| calibrated | 0.797 | 61851 | $0.0124 |
| current_router | 0.781 | 53157 | $0.0106 |
| accuracy_first | 0.778 | 86438 | $0.0173 |
| balanced | 0.772 | 59649 | $0.0119 |
| always_local | 0.753 | 0 | $0.0000 |
| always_remote | 0.744 | 87487 | $0.0175 |

## Limitations

- Synthetic data only; re-run with real models for production calibration.
- Small sample sizes; statistical significance not claimed.
- Profiles with < 10 samples are marked low-confidence.

