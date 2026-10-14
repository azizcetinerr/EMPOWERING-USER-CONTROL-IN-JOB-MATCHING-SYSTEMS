# MMR (Maximal Marginal Relevance) Algorithm Integration and Troubleshooting Report

## 📋 Contents
1. [Project Summary](#1-project-summary)
2. [Initial State and Problems Detected](#2-initial-state-and-problems-detected)
3. [MMR Algorithm Integration](#3-mmr-algorithm-integration)
4. [Technical Problems Encountered and Their Solutions](#4-technical-problems-encountered-and-their-solutions)
5. [Tests Performed and Results](#5-tests-performed-and-results)
6. [Current Status](#6-current-status)
7. [Future Suggestions](#7-future-suggestions)
8. [Technical Details](#8-technical-details)

---

## 1. Project Summary

This project is a TUBITAK-supported recommender system development effort. The system recommends suitable projects based on candidates' CVs. The system consists of the following components:

- **Matching Score (M-Score)**: fit score between the candidate's skills and the project requirements
- **Sincerity Score (S-Score)**: the candidate's sincerity score
- **Reliability Score (R-Score)**: reliability score of the project information
- **Final Ranking**: final ranking with the weighted mean of all scores
- **Performance Metrics**: Relevance Accuracy, Diversity Score, Confidence, Recommendation Coverage, User Coverage

### Purpose of the system
Besides recommending the most suitable projects to candidates, the system should also ensure that the recommended projects differ from each other (are diverse), because recommending projects of the same kind lowers the user experience and reduces the effectiveness of the system.

---

## 2. Initial State and Problems Detected

### 2.1. First problem detected: "mathematical cloning"

**Problem definition:**
When the system metrics were computed, the following results were obtained:

```json
{
  "metrics": {
    "relevance_accuracy": 52.9798,
    "diversity_score": 0.0,
    "confidence": 0.0586,
    "recommendation_coverage": 0.0053,
    "user_coverage": 0.0054
  }
}
```

**Symptoms of the problem:**
1. **Diversity Score = 0.0**: this showed that the recommended projects were almost identical to each other
2. **Confidence = 0.0586**: a very low confidence score, showing that the scores were very close to each other (low standard deviation)
3. Together, these two observations were called the "mathematical cloning" problem

**Causes of the problem:**
- The system ranked only by the relevance score
- The diversity factor was not considered at all
- Since the highest-scoring projects were similar to each other, the top-5 recommendations looked like copies of each other

### 2.2. First solution attempts

#### Attempt 1: Fixing the Confidence computation
- **Done**: raw scores started to be used instead of normalized scores in the Confidence metric computation
- **Result**: the Confidence value changed slightly (0.0586 → 0.0429) but the Diversity Score stayed at 0.0
- **Why it failed**: the problem was not in confidence but in diversity

#### Attempt 2: Logging and debugging
- **Done**: detailed logging was added to the diversity computation code
- **Result**: the source of the problem was better understood but no solution was achieved
- **Why it failed**: logging showed the problem but did not solve it

---

## 3. MMR Algorithm Integration

### 3.1. What is the MMR algorithm?

The **MMR (Maximal Marginal Relevance)** algorithm is a re-ranking algorithm used in information retrieval and recommender systems.

**Goal**: to achieve the optimal balance between relevance and diversity.

**Formula**:
```
MMR_Score = λ * Relevance + (1 - λ) * Diversity
```

Where:
- **λ (lambda)**: relevance weight (between 0.0 and 1.0)
  - λ = 1.0 → only relevance matters (no diversity)
  - λ = 0.5 → relevance and diversity have equal weight
  - λ = 0.0 → only diversity matters (no relevance)
- **Relevance**: the project's fit with the candidate profile (Fair_Score)
- **Diversity**: how different the project is from the previously selected projects (1 - maximum cosine similarity)

### 3.2. How the MMR algorithm works

1. **First project selection**: the project with the highest Fair_Score is selected (already first in the sorted list)

2. **Iteration over the remaining projects**:
   - For each remaining project:
     - **Relevance**: Fair_Score (normalized: between 0 and 1)
     - **Diversity**: the complement of the maximum cosine similarity to the selected projects (1 - max_similarity)
     - **MMR_Score**: λ * Relevance + (1 - λ) * Diversity
   - The project with the highest MMR_Score is selected
   - The selected project is added to the "selected projects" list

3. **Result**: the Top-K projects are selected with the balance of relevance and diversity in mind

### 3.3. MMR integration - code changes

#### 3.3.1. MMR settings added to the config file

**File**: `config/config.yaml`

```yaml
# Diversity-Aware Ranking (MMR) Settings
diversity_aware_ranking:
  enabled: true  # Enable the MMR algorithm
  top_k: 10  # Number of projects selected with MMR (top-K)
  lambda: 0.3  # Relevance weight (0.0-1.0), 0.3 = 30% relevance, 70% diversity
```

**Why these values?**
- `enabled: true`: to activate MMR
- `top_k: 10`: select the first 10 projects with MMR (a sufficient number for more diversity)
- `lambda: 0.3`: to increase diversity we gave it more weight than relevance (70% diversity, 30% relevance)

#### 3.3.2. MMR method added to the FinalRankingCalculator class

**File**: `src/scoring/final_ranking.py`

**Changes made**:

1. **MMR config loading added to the `__init__` method**:
```python
# MMR (Diversity-Aware Ranking) settings
mmr_config = config.get('diversity_aware_ranking', {})
self.mmr_enabled = mmr_config.get('enabled', False)
self.mmr_top_k = mmr_config.get('top_k', 10)
self.mmr_lambda = mmr_config.get('lambda', 0.5)
```

2. **MMR call added to the `calculate_ranking` method**:
```python
# Sort
df_result = df_result.sort_values(by='Fair_Score', ascending=False).reset_index(drop=True)

# Diversity-aware top-K selection with MMR (Maximal Marginal Relevance)
if self.mmr_enabled and self.mmr_top_k > 0:
    self.logger.info(f"🔄 Applying diversity-aware ranking with MMR (Maximal Marginal Relevance) (Top-{self.mmr_top_k})...")
    df_result = self._apply_mmr_ranking(
        df_result,
        top_k=self.mmr_top_k,
        lambda_param=self.mmr_lambda
    )
```

3. **`_apply_mmr_ranking` method added**:
   - Selects the first project (highest Fair_Score)
   - Computes the MMR score for the remaining projects
   - Selects the projects with the highest MMR scores
   - Moves the selected projects to the top and keeps the rest in the original order

**Why this approach?**
- Selecting the Top-K projects with MMR increases diversity
- Keeping the remaining projects in the original order preserves the overall performance of the system
- Only the diversity of the recommended projects is increased; the whole system is not affected

---

## 4. Technical Problems Encountered and Their Solutions

### 4.1. Problem: inconsistent vector formats

**Problem definition**:
The vectors in the `skill_vectors` column were in different formats:
- Some were 1D arrays: `(768,)`
- Some were 2D arrays: `(1, 768)` or `(7, 768)` or `(9, 768)`
- Some were scalar floats (NaN)

**Error message**:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 1, the array at index 0 has size 768 and the array at index 1 has size 5376
```

**Why did this problem occur?**
- Different projects have different numbers of skills
- A separate embedding vector was created for each skill
- Some projects have 1 skill → `(1, 768)` format
- Some projects have 7 skills → `(7, 768)` format
- All vectors must have the same size to compute cosine similarity

**Solution**:
We converted all vectors to a standard format (1D array, 768-dimensional):

```python
# 2D vector (n_skills, 768) → 1D vector (768,)
if vec_array.ndim == 2:
    if vec_array.shape[1] == 768:
        normalized_vec = vec_array.mean(axis=0)  # Take the mean → (768,)
    else:
        normalized_vec = vec_array.flatten()
elif vec_array.ndim == 1:
    normalized_vec = vec_array  # Already 1D
elif vec_array.ndim > 2:
    normalized_vec = vec_array.flatten()  # 3D+ → flatten
```

**Why take the mean?**
- If a project has several skills, there is a separate embedding for each skill
- By taking the mean of these embeddings, we create the project's "representative vector"
- This represents the overall skill profile of the project

### 4.2. Problem: pd.isna() does not work on NumPy arrays

**Problem definition**:
```python
if pd.isna(vector):  # Raises an error!
    continue
```

**Error message**:
```
ValueError: The truth value of an array with more than one element is ambiguous. 
Use a.any() or a.all()
```

**Why did this problem occur?**
- `pd.isna()` returns a boolean array for NumPy arrays, not a single boolean
- A boolean array cannot be used in an `if` statement (ambiguous)

**Solution**:
```python
# None check first
if vector is None:
    continue

# NaN float check (can be a NaN float after a pandas merge)
if isinstance(vector, float) and (np.isnan(vector) or pd.isna(vector)):
    continue

# We do not use pd.isna() to check NumPy arrays; we convert directly with np.array()
```

### 4.3. Problem: double normalization

**Problem definition**:
The vectors were normalized twice:
1. In the first loop: when added to `vectors_list` (as normalized)
2. In the second loop: normalized again when added to `normalized_vectors`

**Result**: the vectors were normalized incorrectly, which broke the similarity computations.

**Solution**:
In the first loop we only converted the vectors to 1D (without normalizing):
```python
# First loop: only convert to 1D (do not normalize)
normalized_vec = vec_array.mean(axis=0)  # (768,)
vectors_list.append(normalized_vec)  # Not normalized yet

# Second loop: normalize (make it a unit vector)
norm = np.linalg.norm(vec_array)
if norm > 1e-8:
    vec_normalized = vec_array / norm  # Unit vector
    normalized_vectors.append(vec_normalized)
```

### 4.4. Problem: indentation error (if-else block structure)

**Problem definition**:
In the diversity computation code, the code after `vectors_matrix` was created was outside the `else` block:

```python
if len(normalized_vectors) < 2:
    diversity_score = 0.0
else:
    vectors_matrix = np.vstack(normalized_vectors)

# This code is outside the else! Error!
similarity_matrix = cosine_similarity(vectors_matrix)
```

**Result**: when `normalized_vectors < 2`, `vectors_matrix` was undefined and an error occurred.

**Solution**:
We moved all similarity computation code into the `else` block:
```python
if len(normalized_vectors) < 2:
    diversity_score = 0.0
else:
    vectors_matrix = np.vstack(normalized_vectors)
    similarity_matrix = cosine_similarity(vectors_matrix)
    # ... other computations ...
    diversity_score = 1.0 - avg_similarity
```

### 4.5. Problem: vectors lost after a DataFrame merge

**Problem definition**:
The test scripts could not read the vectors from the DataFrame because:
- The vectors were stored in DataFrame format in the pickle file
- The CSV file did not contain the vectors
- The link formats did not match during the merge

**Solution**:
```python
# Convert the links to strings (to avoid mismatches)
vectors_df['link'] = vectors_df['link'].astype(str)
df['link'] = df['link'].astype(str)
df = df.merge(vectors_df, on='link', how='left')
```

---

## 5. Tests Performed and Results

### 5.1. Test scripts created

#### 5.1.1. `test_mmr.py`
**Purpose**: to test whether MMR works with the current lambda value

**What it does**:
1. Loads the CSV and vector files
2. Computes the baseline metrics (before MMR)
3. Applies MMR
4. Computes the metrics after MMR
5. Compares them

**Result**: MMR works but the Diversity Score is still low

#### 5.1.2. `test_mmr_lambda.py`
**Purpose**: to test different lambda values and find the best one

**What it does**:
1. Tests the lambda values: [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
2. Applies MMR for each lambda
3. Computes the Diversity and Relevance metrics
4. Recommends the best lambda value

**Results**:
```
Lambda     Diversity    Relevance    Change
Baseline   0.0028       52.98        -
0.9        0.0000       52.98        -0.0028
0.7        0.0000       52.98        -0.0028
...
0.0        0.0000       52.98        -0.0028
```

**Observation**: no lambda value increased the Diversity Score. This shows that after MMR the vectors could not be processed in the diversity computation code.

#### 5.1.3. `debug_diversity.py` (one-off script, no longer in the repository)
**Purpose**: to analyze in detail why the Diversity Score is 0.0

**What it does**:
1. Inspects the vectors of the top-5 projects
2. Checks the type, shape and dtype of each vector
3. Computes the pairwise similarities
4. Provides detailed debug information

**Result**: the inconsistency of the vector formats was detected

### 5.2. Test results

#### Baseline (before MMR)
```
Diversity Score: 0.0028
Relevance Accuracy: 52.98
Confidence: 0.0429
```

**Comment**:
- Diversity is very low (almost 0)
- This shows that the top-5 projects are very similar to each other
- 3/5 projects have vectors (2 projects have NaN)

#### After MMR (lambda=0.0 - diversity only)
```
Diversity Score: 0.0000
Relevance Accuracy: 52.98
```

**Comment**:
- MMR works (4/5 project positions changed)
- But the Diversity Score is still 0.0
- This shows that there is a problem in the diversity computation code
- After MMR, all projects in the top-5 have vectors

---

## 6. Current Status

### 6.1. Improvements achieved

✅ **MMR algorithm integrated**:
- `_apply_mmr_ranking` method added
- Settings added to the config file
- Integrated into the `calculate_ranking` method

✅ **Vector processing fixed**:
- 2D vectors → converted to 1D (with the mean)
- NaN float check added
- `pd.isna()` problem solved
- Double normalization problem solved

✅ **Code errors fixed**:
- Indentation error fixed
- Exception handling improved
- Vector processing logic simplified

✅ **Test infrastructure created**:
- `test_mmr.py`: basic MMR test
- `test_mmr_lambda.py`: lambda parameter optimization
- `debug_diversity.py`: detailed diversity analysis (one-off script, no longer in the repository)

### 6.2. Remaining problems

⚠️ **The Diversity Score is still 0.0 after MMR**:
- **Problem**: after MMR all projects in the top-5 have vectors, but only 1 vector is processed in the diversity computation code
- **Error message**: "⚠️ Could not compute diversity: not enough vectors (1/5)"
- **Possible causes**:
  1. The references to the vectors may have been lost after `pd.concat`
  2. The exception handling is too broad; vectors may be skipped silently
  3. An edge case may have been missed in the vector processing code

⚠️ **The baseline diversity is very low**:
- Diversity = 0.0028 at baseline (almost 0)
- This shows that the top-5 projects really are very similar
- Diversity is low even without MMR

### 6.3. Code status

**Working code**:
- ✅ MMR algorithm implementation
- ✅ Vector normalization logic
- ✅ Baseline diversity computation
- ✅ Test scripts

**Problematic code**:
- ⚠️ Diversity computation after MMR (vectors cannot be processed)

---

## 7. Future Suggestions

### 7.1. Short-term solutions

#### Suggestion 1: debug the vector processing code in detail
```python
# Add detailed logging to the diversity computation code
for idx, row in top_projects.iterrows():
    vector = row[vector_column]
    self.logger.debug(f"Project {idx}: vector type={type(vector)}, shape={getattr(vector, 'shape', 'N/A')}")
    # ... vector processing ...
    self.logger.debug(f"Project {idx}: vectors_list length={len(vectors_list)}")
```

#### Suggestion 2: narrow the exception handling
```python
# Catch specific exceptions instead of a broad exception
except ValueError as e:
    self.logger.warning(f"Vector processing error (ValueError): {e}")
except TypeError as e:
    self.logger.warning(f"Vector processing error (TypeError): {e}")
except AttributeError as e:
    self.logger.warning(f"Vector processing error (AttributeError): {e}")
```

#### Suggestion 3: store the vectors directly in the DataFrame
So that the vectors are not lost after MMR, always reload them from the pickle file instead of keeping them directly in the DataFrame:

```python
# Reload the vectors after MMR
df_mmr = ranking_calc._apply_mmr_ranking(...)
# Merge the vectors again
df_mmr = df_mmr.merge(vectors_df[['link', 'skill_vectors']], on='link', how='left')
```

### 7.2. Medium-term improvements

#### Suggestion 1: standardize the vector format
Convert all vectors to a standard format in the preprocessing stage:
- Convert all vectors to the 1D array (768,) format
- Remove NaN vectors or use a default vector

#### Suggestion 2: try alternative diversity metrics
- **Intra-List Diversity (ILD)**: diversity within the recommended list
- **Coverage-based diversity**: recommending projects from different categories
- **Content-based diversity**: diversity by project content

#### Suggestion 3: make the lambda parameter dynamic
Adjust lambda according to user preferences:
- If the user wants "more diversity" → lower lambda
- If the user wants "better-fitting projects" → raise lambda

### 7.3. Long-term improvements

#### Suggestion 1: improve the embedding quality
- Use a better embedding model
- Do domain-specific fine-tuning
- Use multi-modal embeddings (text + metadata)

#### Suggestion 2: hybrid recommender system
- Content-based filtering + collaborative filtering
- MMR + clustering (category-based diversity)
- Dynamic weighting based on user feedback

#### Suggestion 3: A/B testing infrastructure
Test different lambda values with real users:
- User satisfaction metrics
- Click-through rate (CTR)
- Engagement metrics

---

## 8. Technical Details

### 8.1. Cosine similarity computation

**Formula**:
```
similarity(A, B) = (A · B) / (||A|| * ||B||)
```

**Use in the code**:
```python
# Normalize the vectors (unit vector)
vec_normalized = vec_array / np.linalg.norm(vec_array)

# Cosine similarity = dot product (for normalized vectors)
similarity = np.dot(vec1_normalized, vec2_normalized)
```

**Why cosine similarity?**
- Independent of the magnitude of the vectors (only the direction matters)
- Very effective for embedding vectors
- Gives a value in the 0-1 range (easy to interpret)

### 8.2. Diversity Score computation

**Formula**:
```
Diversity = 1 - Mean_Cosine_Similarity
```

**Use in the code**:
```python
# Pairwise similarity matrix
similarity_matrix = cosine_similarity(vectors_matrix)

# Set the diagonal to 0 (self-similarity = 1, which we do not want)
np.fill_diagonal(similarity_matrix, 0.0)

# Take the mean of the upper triangle (to count each pair once)
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
avg_similarity = similarity_matrix[mask].mean()

# Diversity = 1 - mean similarity
diversity_score = 1.0 - avg_similarity
```

**Why the upper triangle?**
- The similarity matrix is symmetric (A-B = B-A)
- To count each pair only once
- Setting the diagonal to 0: self-similarity = 1, which is meaningless for diversity

### 8.3. Detailed description of the MMR algorithm

**Pseudo-code**:
```
1. selected = [first_project]  # Highest score
2. remaining = all_projects - selected
3. WHILE len(selected) < top_k:
    4. best_score = -inf
    5. best_project = None
    6. FOR each project in remaining:
        7. relevance = project.score / max_score
        8. diversity = 1 - max(similarity(project, s) for s in selected)
        9. mmr_score = lambda * relevance + (1-lambda) * diversity
        10. IF mmr_score > best_score:
            11. best_score = mmr_score
            12. best_project = project
    13. selected.append(best_project)
    14. remaining.remove(best_project)
15. RETURN selected + remaining (in original order)
```

**Time complexity**:
- O(k * n * d) where:
  - k = top_k (number of projects to select)
  - n = total number of projects
  - d = embedding dimension (768)
- In each iteration, the similarity is computed for all remaining projects

**Optimization suggestions**:
- Use vectorized operations (NumPy)
- Use batch processing
- Early stopping (stop when sufficient diversity is reached)

### 8.4. Vector processing pipeline

```
1. Vector loading
   ├─ Read from the pickle file
   ├─ In DataFrame format? → Merge
   └─ In dict format? → Add with apply

2. Standardizing the vector format
   ├─ 2D array (n, 768)? → mean(axis=0) → (768,)
   ├─ 1D array (768,)? → Use directly
   ├─ 3D+ array? → flatten()
   └─ Scalar/NaN? → Skip

3. Size check
   ├─ > 768? → [:768] (truncate)
   ├─ < 768? → Padding (zeros)
   └─ = 768? → OK

4. Normalization
   ├─ Compute the norm
   ├─ Norm > 1e-8? → Normalize (unit vector)
   └─ Norm <= 1e-8? → Skip (zero vector)

5. Similarity computation
   ├─ Normalized vectors → cosine similarity
   └─ Diversity = 1 - mean similarity
```

---

## 9. Conclusion and Summary

### 9.1. What was done

1. ✅ **MMR algorithm integrated**: the MMR algorithm was added for diversity-aware ranking
2. ✅ **Vector processing fixed**: inconsistent vector formats were standardized
3. ✅ **Code errors fixed**: indentation, exception handling and normalization problems were solved
4. ✅ **Test infrastructure created**: 3 different test scripts were written

### 9.2. Remaining work

1. ⚠️ **Diversity computation after MMR**: find and fix why the vectors cannot be processed
2. ⚠️ **Improving the baseline diversity**: increase diversity even without MMR
3. 📝 **Lambda optimization**: find the best lambda value (currently being tested but without results)

### 9.3. Lessons learned

1. **Vector formats must be consistent**: vectors in different formats cause problems
2. **Exception handling must be careful**: overly broad exception handling can hide real problems
3. **Test infrastructure matters**: without detailed test scripts it is very hard to find problems
4. **The diversity computation is sensitive**: vector normalization, similarity computation, every step is critical

### 9.4. Recommendations

1. **Short term**: detailed debugging should be done to solve the diversity computation problem after MMR
2. **Medium term**: strengthen the vector preprocessing pipeline and standardize all vectors from the start
3. **Long term**: a better embedding model, a hybrid recommender system, an A/B testing infrastructure

---

## 10. Additional Information

### 10.1. Technologies used

- **Python 3.x**
- **NumPy**: vector operations, mathematical computations
- **Pandas**: DataFrame operations
- **scikit-learn**: cosine similarity computation
- **XLM-RoBERTa**: embedding model (768-dimensional)

### 10.2. File structure

```
EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS/
├── src/
│   └── scoring/
│       └── final_ranking.py          # MMR implementation
├── config/
│   └── config.yaml                   # MMR settings
├── tests/test_mmr.py                 # Basic MMR test
├── tests/test_mmr_lambda.py          # Lambda optimization
└── IP1_output_listings_vectorized.pkl # Vector file
```

### 10.3. References

- **MMR algorithm**: Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries.
- **Diversity metrics**: Vargas, S., & Castells, P. (2011). Rank and relevance in novelty and diversity metrics for recommendation.
- **Cosine similarity**: scikit-learn documentation

---

**Report date**: 31 December 2025  
**Prepared by**: AI Assistant (Cursor)  
**Project**: TUBITAK Recommender System - MMR Integration
