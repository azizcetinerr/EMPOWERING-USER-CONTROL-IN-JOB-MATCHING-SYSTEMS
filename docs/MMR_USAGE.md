# Using MMR (Maximal Marginal Relevance)

## Status
The MMR (Maximal Marginal Relevance) algorithm was added and enabled. This algorithm adds a diversity factor to the ranking to increase the Diversity Score.

## Important Note ⚠️

**Your current metric results were probably produced without MMR!** To see the effect of MMR:

1. **Rerun the pipeline** (Flask app or `src/pipeline.py`)
2. **OR** run the `tests/test_mmr.py` script to test whether MMR works

## Config Settings

In `config/config.yaml`:

```yaml
diversity_aware_ranking:
  enabled: true   # MMR active
  top_k: 10       # Number of projects selected with MMR
  lambda: 0.6     # Relevance weight (0.6 = 60% relevance, 40% diversity)
```

### Setting the lambda parameter

- **lambda = 1.0**: relevance only (no diversity)
- **lambda = 0.5**: equal weight (relevance and diversity)
- **lambda = 0.0**: diversity only (no relevance)

**To increase the Diversity Score, try lowering lambda:**
- `lambda: 0.4` → 60% diversity, 40% relevance
- `lambda: 0.3` → 70% diversity, 30% relevance

## Test Script

To test whether MMR works:

```bash
python tests/test_mmr.py
```

This script:
- Checks the MMR settings
- Loads the existing results
- Compares the metrics before and after MMR
- Shows whether the Diversity Score increased

## How Does MMR Work?

1. **First project**: the project with the highest Fair_Score is selected
2. **Next projects**: at each step, the best remaining project is selected with this formula:
   ```
   MMR_score = lambda * relevance + (1 - lambda) * diversity
   ```
   - **relevance**: Fair_Score (normalized to 0-1)
   - **diversity**: 1 - maximum similarity to the selected projects (1 - max_similarity)

3. **Top-K**: the first K projects are selected with MMR, the rest keep the original order

## Expected Results

With MMR active:
- **The Diversity Score should increase** (rise above 0.0)
- **Relevance Accuracy may drop slightly** (trade-off for diversity)
- **Confidence may change** (because different projects are selected)

## Troubleshooting

### Diversity Score is still 0.0

1. **Rerun the pipeline** (old results may be without MMR)
2. **Lower lambda** (e.g. 0.4 or 0.3)
3. **Check that the vectors exist** (`IP1_output_listings_vectorized.pkl`)
4. **Check the log messages** (is MMR running?)

### MMR does not run

1. Make sure `diversity_aware_ranking.enabled: true` is set in `config.yaml`
2. Look for the message "🔄 Applying diversity-aware ranking with MMR (Maximal Marginal Relevance)" in the logs
3. Run the `tests/test_mmr.py` script

## Example Usage

1. **Set the config**:
   ```yaml
   diversity_aware_ranking:
     enabled: true
     top_k: 10
     lambda: 0.5  # Or lower (0.4, 0.3)
   ```

2. **Run the pipeline**:
   ```bash
   python src/pipeline.py
   # OR
   python web/app.py  # Flask app
   ```

3. **Check the results**:
   - Look for the message "MMR ranking completed" in the log file
   - Check the metric reports in the `outputs/metrics/` directory
   - See whether the Diversity Score increased
