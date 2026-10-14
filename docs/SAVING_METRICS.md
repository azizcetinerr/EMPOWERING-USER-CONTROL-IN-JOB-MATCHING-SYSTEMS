# Saving the Metrics

To compute and save the metrics from existing IP5 results:

## Usage

```bash
python scripts/save_metrics_from_existing.py
```

or

```bash
python3 scripts/save_metrics_from_existing.py
```

## What Does the Script Do?

1. Loads `outputs/results/IP5_final_fair_ranking.csv`
2. Loads the vectors from `IP1_output_listings_vectorized.pkl` (needed for diversity)
3. Computes 5 metrics:
   - Relevance Accuracy (mean M_Score)
   - Diversity Score
   - Confidence (std deviation of the M scores)
   - RC (Recommendation Coverage)
   - UC (User Coverage)
4. Saves the results to `outputs/metrics/`:
   - `metrics_report_YYYYMMDD_HHMMSS.json` (JSON format)
   - `metrics_report_YYYYMMDD_HHMMSS.md` (Markdown format)
   - `metrics_visualization_YYYYMMDD_HHMMSS.png` (chart)

## Note

The script must use the same Python interpreter as your web application.
