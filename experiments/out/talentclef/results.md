# TalentCLEF 2026 Task A (English, development set): external validation

Queries (job postings): 10; CVs: 472; relevant pairs: 472 (binary labels). Encoder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2; tau 0.165 / 0.783 from sanity_esco.json. Nothing tuned on TalentCLEF.

## Main table: every query ranks all CVs

| System | NDCG@10 | MAP | P@10 | Recall@100 |
|---|---|---|---|---|
| BM25 | 0.894 | 0.857 | 0.880 | 0.939 |
| Dense | 0.828 | 0.774 | 0.830 | 0.904 |
| M_old | 0.732 | 0.575 | 0.700 | 0.647 |
| M_meanmax | 0.903 | 0.793 | 0.900 | 0.913 |
| M | 0.878 | 0.772 | 0.880 | 0.897 |
| E | 0.670 | 0.559 | 0.630 | 0.739 |
| M+E | 0.911 | 0.779 | 0.890 | 0.897 |

Paired comparisons over queries (randomization test, 10,000 sign flips; bootstrap 95% CI; Holm correction over the three comparisons per metric):

| Metric | Comparison | Mean difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| ndcg@10 | M vs BM25 | -0.016 | [-0.072, +0.035] | 0.6233 | 1.0000 |
| ndcg@10 | M vs Dense | +0.049 | [-0.053, +0.168] | 0.4349 | 1.0000 |
| ndcg@10 | M vs M_old | +0.146 | [-0.035, +0.371] | 0.3485 | 1.0000 |
| map | M vs BM25 | -0.085 | [-0.131, -0.042] | 0.0087 | 0.0261 |
| map | M vs Dense | -0.002 | [-0.089, +0.089] | 0.9631 | 0.9631 |
| map | M vs M_old | +0.197 | [+0.043, +0.389] | 0.0359 | 0.0718 |
| p@10 | M vs BM25 | -0.000 | [-0.040, +0.050] | 1.0000 | 1.0000 |
| p@10 | M vs Dense | +0.050 | [-0.010, +0.120] | 0.3751 | 0.7501 |
| p@10 | M vs M_old | +0.180 | [+0.010, +0.400] | 0.1256 | 0.3768 |
| recall@100 | M vs BM25 | -0.043 | [-0.071, -0.019] | 0.0172 | 0.0344 |
| recall@100 | M vs Dense | -0.008 | [-0.044, +0.044] | 0.8305 | 0.8305 |
| recall@100 | M vs M_old | +0.249 | [+0.084, +0.442] | 0.0054 | 0.0162 |

## Additional table: candidate set = labeled CVs + BM25 top 100

| System | NDCG@10 | MAP | P@10 | Recall@100 |
|---|---|---|---|---|
| BM25 | 0.894 | 0.869 | 0.880 | 0.939 |
| Dense | 0.837 | 0.819 | 0.840 | 0.962 |
| M_old | 0.837 | 0.728 | 0.790 | 0.967 |
| M_meanmax | 0.903 | 0.829 | 0.900 | 0.972 |
| M | 0.885 | 0.816 | 0.890 | 0.971 |
| E | 0.767 | 0.743 | 0.730 | 0.975 |
| M+E | 0.911 | 0.822 | 0.890 | 0.971 |

Paired comparisons over queries (randomization test, 10,000 sign flips; bootstrap 95% CI; Holm correction over the three comparisons per metric):

| Metric | Comparison | Mean difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| ndcg@10 | M vs BM25 | -0.009 | [-0.053, +0.037] | 0.6860 | 1.0000 |
| ndcg@10 | M vs Dense | +0.048 | [-0.047, +0.164] | 0.4964 | 1.0000 |
| ndcg@10 | M vs M_old | +0.047 | [-0.061, +0.190] | 0.6894 | 1.0000 |
| map | M vs BM25 | -0.053 | [-0.094, -0.015] | 0.0540 | 0.1620 |
| map | M vs Dense | -0.003 | [-0.073, +0.076] | 0.9377 | 0.9377 |
| map | M vs M_old | +0.088 | [-0.003, +0.213] | 0.1451 | 0.2902 |
| p@10 | M vs BM25 | +0.010 | [-0.030, +0.060] | 1.0000 | 1.0000 |
| p@10 | M vs Dense | +0.050 | [-0.010, +0.120] | 0.3729 | 0.7640 |
| p@10 | M vs M_old | +0.100 | [+0.000, +0.250] | 0.2547 | 0.7640 |
| recall@100 | M vs BM25 | +0.032 | [+0.010, +0.056] | 0.0659 | 0.1977 |
| recall@100 | M vs Dense | +0.009 | [-0.002, +0.024] | 0.3793 | 0.7585 |
| recall@100 | M vs M_old | +0.004 | [-0.007, +0.019] | 1.0000 | 1.0000 |

Runtime 7614 s; python 3.11.0, macOS-27.0.1-arm64-arm-64bit
