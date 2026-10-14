# Performance Metrics Module - Documentation

## Summary

To measure the recommender's performance, a `calculate_performance_metrics` function was added to the `FinalRankingCalculator` class. This function computes 3 metrics for the best 5 projects (Top-K):

1. **Relevance Accuracy**: mean of M_Score
2. **Diversity Score**: 1 - mean similarity, from item-to-item cosine similarity
3. **Confidence**: standard deviation of the M scores

## Changes Made

### 1. `src/scoring/final_ranking.py`

#### a) Imports added
```python
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
```

#### b) New function: `calculate_performance_metrics`
- **Location**: inside the `FinalRankingCalculator` class
- **Parameters**:
  - `df_ranked`: ranked project DataFrame
  - `top_k`: number of best projects to evaluate (default: 5)
  - `vector_column`: column holding the project vectors (default: "skill_vectors")
  - `m_score_column`: Matching Score column (default: "M_Score")
- **Return**: `(relevance_accuracy, diversity_score, confidence)` tuple

#### c) Integration
Called automatically at the end of the `calculate_ranking` function (after the ranking is complete).

## Metric Details

### 1. Relevance Accuracy
- **Formula**: `mean(M_Score)` (for the Top-K projects)
- **Range**: 0-100 (for normalized scores)
- **Description**: shows how high a match the system can find for the user's CV
- **Higher = better**: a higher mean M score means more relevant recommendations

### 2. Diversity Score (ILS - Intra-List Similarity)
- **Formula**: `1 - mean(cosine_similarity(pairwise))`
- **Range**: 0-1
- **Description**: measures how similar the recommended projects are to each other
- **Higher = better**: higher diversity means more varied recommendations
- **Computation**:
  1. Take the vectors of the Top-K projects
  2. Compute the pairwise cosine similarity
  3. Set the diagonal to 0 (excluding self-similarity)
  4. Take the mean of the upper triangle
  5. Diversity = 1 - mean similarity

### 3. Confidence
- **Formula**: `std(M_Score)` (for the Top-K projects)
- **Range**: 0+ (standard deviation)
- **Description**: measures the consistency of the M scores of the recommended projects
- **Interpretation**:
  - Low std: consistent recommendations (similar quality)
  - High std: recommendations of varying quality

## Output Format

The function is called automatically inside `calculate_ranking` and the results are written by the logger:

```
📊 SYSTEM PERFORMANCE: Relevance=75.32 | Diversity=0.68 | Confidence=12.45
```

## Error Handling

The function handles errors gracefully:
- No vectors: Diversity = 0.0
- No M_Score: Relevance = 0.0, Confidence = 0.0
- Single project: Diversity = 0.0 (diversity cannot be computed)
- Invalid vector format: a warning is logged, Diversity = 0.0

## Usage Example

The function is called automatically inside `calculate_ranking`:

```python
# Inside the pipeline
ranking_calc = FinalRankingCalculator(config, logger)
df_final = ranking_calc.calculate_ranking(df_projects)
# The metrics are computed and logged automatically
```

Manual use (optional):

```python
relevance, diversity, confidence = ranking_calc.calculate_performance_metrics(
    df_final,
    top_k=5,
    vector_column="skill_vectors",
    m_score_column="M_Score"
)
```

## Notes

- Cosine similarity is computed on normalized vectors
- The Top-K parameter can be changed (default: 5)
- Only projects with valid vectors are included in the diversity computation
- The performance impact is minimal (computed only for the Top-K)
