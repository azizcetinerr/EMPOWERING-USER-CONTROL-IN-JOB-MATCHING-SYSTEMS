# Lambda Test Instructions

## Problem
The MMR Diversity Score stays at 0.0 and does not change.

## Solution Steps

### 1. Run the lambda test script

```bash
python tests/test_mmr_lambda.py
```

This script:
- Tests 8 different lambda values (0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
- Computes the Diversity Score for each lambda
- Finds and recommends the best lambda value

### 2. Add the best lambda value to the config

Find the lambda value marked "⭐ BEST" in the script output and add it to config.yaml:

```yaml
diversity_aware_ranking:
  enabled: true
  top_k: 10
  lambda: 0.1  # Best value found by the script
```

### 3. Rerun the pipeline

Run the pipeline with the new lambda value.

## Notes

- Lambda = 0.0 → diversity only (no relevance)
- Lambda = 1.0 → relevance only (no diversity)
- Lambda = 0.3 → 30% relevance, 70% diversity
- Lambda = 0.1 → 10% relevance, 90% diversity

If no lambda increases the Diversity Score, the projects really are very similar. In that case:
- Try increasing top_k (e.g. 20 or 30)
- Or accept that the project pool really is very similar
