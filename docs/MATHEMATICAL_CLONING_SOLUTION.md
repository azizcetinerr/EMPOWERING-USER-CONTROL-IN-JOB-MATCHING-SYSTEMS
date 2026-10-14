# Mathematical Cloning Problem - Suggested Solutions

## Problem

- **Diversity Score: 0.0** → the top 5 projects are very similar to each other
- **Confidence: 0.0586** → the M_Score values are almost identical (very low std deviation)

## Main Cause

**Z-Score normalization** pulls all scores towards 50, which:
1. Brings the M_Score values of all projects close to each other
2. Lowers the standard deviation (low Confidence)
3. Makes the ranking depend only on small differences

## Solution 1: Normalization improvement (recommended)

Use a wider range in the Z-Score normalization:

```python
# Current: normalized = 50 + (z_scores * 20)  # ±2.5 std = 0-100
# Suggested: normalized = 50 + (z_scores * 25)  # ±2 std = 0-100 (wider range)
```

## Solution 2: Diversity-aware ranking (most effective)

Add a diversity factor to the top-K selection:

```python
def diversity_aware_ranking(df, top_k=5, lambda_param=0.5):
    """
    MMR (Maximal Marginal Relevance)-like ranking with a diversity factor
    
    Args:
        df: Project DataFrame (sorted by Fair_Score)
        top_k: Number of projects to select
        lambda_param: Balance between relevance (0.0) and diversity (1.0) (0.5 = equal)
    
    Returns:
        Top-k projects with the diversity factor
    """
    selected = []
    remaining = df.copy()
    
    # First project: the highest score
    first = remaining.nlargest(1, 'Fair_Score')
    selected.append(first.index[0])
    remaining = remaining.drop(first.index[0])
    
    # Selection with the diversity factor for the remaining projects
    while len(selected) < top_k and len(remaining) > 0:
        max_score = -1
        best_idx = None
        
        for idx, row in remaining.iterrows():
            # Relevance score (Fair_Score)
            relevance = row['Fair_Score']
            
            # Diversity score (similarity to the selected projects)
            diversity = calculate_diversity_score(row, selected, df)
            
            # Combined score: lambda * relevance + (1-lambda) * diversity
            combined = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if combined > max_score:
                max_score = combined
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining = remaining.drop(best_idx)
    
    return df.loc[selected]
```

## Solution 3: Use raw scores in the metric computation

Use raw scores instead of normalized scores for the Confidence metric:

```python
# Current: m_score_column="M_Score" (normalized)
# Suggested: m_score_column="M_Score_Raw" (raw scores)
```

## Solution 4: Increase Top-K

Show a wider range by increasing Top-K from 5 to 10.

## Implementation Priority

1. **Solution 3** (fastest): use raw scores in the metrics
2. **Solution 1** (medium): widen the normalization range
3. **Solution 2** (most effective but complex): add diversity-aware ranking
