# Mathematical Cloning Problem Analysis

## Current Situation

A serious "mathematical cloning" problem was detected in the metrics:

- **Diversity Score**: 0.0 (no diversity among the top 5 projects)
- **Confidence**: 0.0586 (the M_Score standard deviation is very low, the scores are almost identical)
- **Relevance Accuracy**: 52.98 (normal)

## Causes of the Problem

1. **Z-Score normalization is too aggressive**
   - Z-Score normalization pulls all scores towards 50
   - Formula: `normalized = 50 + (z_score * 20)`
   - This makes all M_Score values very close to each other

2. **Vector similarity**
   - The vectors of the top 5 projects are very similar
   - High cosine similarity → low diversity

3. **Ranking algorithm**
   - Ranking is based only on the scores
   - The diversity factor is ignored

## Suggested Solutions

### 1. Diversity-aware ranking
- A diversity factor should be added to the top-K selection
- The MMR (Maximal Marginal Relevance) algorithm can be used
- Combination of score + diversity bonus

### 2. Normalization improvement
- Min-Max normalization instead of Z-Score normalization
- Or a wider range for the Z-Score (current: ±2.5 std, suggested: ±3 std)

### 3. Vector diversity check
- Integrate the Diversity Score into the ranking
- Filter similar projects or add a diversity bonus
