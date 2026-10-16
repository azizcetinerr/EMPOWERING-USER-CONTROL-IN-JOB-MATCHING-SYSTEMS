# Corrected-system experiments

Re-scores the December 2025 snapshot (939 distinct listings) with the corrected ranker,
compares it with baselines and with the reported run, and evaluates every system against
graded relevance labels for the one candidate.

## What changed relative to the reported run

| Component | Reported run | Corrected |
|---|---|---|
| Encoder | XLM-R, mean over all positions incl. padding | chosen by ROC-AUC on 100 labeled skill pairs (`skill_pairs.csv`) among XLM-R (masked / unmasked), multilingual MPNet, LaBSE, multilingual-E5 |
| M | mean of best cosine per listing skill | IDF-weighted soft coverage; thresholds = median cosine of unrelated / related pairs |
| S → L | AI-detector constant + letter–listing cosine | letter–listing cosine only (detector removed); called *letter–listing relevance* |
| R → E | logistic of the entailment logit, first 10 CV lines, host bonuses | softmax entailment probability, all CV paragraphs, IDF-weighted; host terms removed (fields are empty); called *CV evidence* |
| Fusion | z-score 50 + 20z, clipped | percentile rank (main), z-score reported as a variant; weights unchanged (0.5/0.3/0.2) |
| Keyword bonus | substring test, 909/952 listings | removed |
| Duplicates | 13 duplicated records | removed by link |
| Listing skills | SkillNER output as stored (on non-English listings mostly function words: "los", "en", "ses") | skills that are a stopword in any of 24 spaCy languages removed; non-English listings keep only skills also extracted from an English listing; listings left without skills get the median M and E (added before labeling) |
| Few-skill listings | M and E are means over 1–3 skills for ~10% of listings, so they take extreme values | M and E shrunk toward the median: (n·score + 5·median)/(n + 5), n = IDF-weighted skill count; unshrunk scores kept (added before labeling) |
| Long descriptions | – (the selected encoder reads 128 tokens; 99% of descriptions are longer, median 489) | 120-token chunks with 20-token overlap: L and dense = best chunk, MMR/ILD = normalized chunk mean; the dense baseline's profile query (CV + letter, ~1,500 tokens) is chunked too and represented by its normalized chunk mean; first-128-token scores kept as `L_trunc128`, `dense_trunc128` |
| MMR | λ = 0.2 on mean skill vectors | off and λ ∈ {0.9, 0.7, 0.5, 0.2} on description embeddings, all reported |

Versions of the fused ranker (both evaluated as separate systems):
- **Fused (v1, pre-registered)** – `config_v1.yaml`: stored skills, no shrinkage (listings without
  skills get 0), L on the first 128 tokens.
- **Fused (v2)**, the main system – `config.yaml`: language filter, shrinkage with k = 5, chunked L;
  k = 0, 2.5 and 10 reported as variants; MMR and the ablations are run on v2.
  The language filter, the shrinkage and the chunking were added before labeling, after inspecting
  the top 10 of v1 and of intermediate versions.

Systems compared: reported run (A and raw M), BM25 / TF-IDF / dense retrieval with the
whole profile (CV + letter) as query, M, L, E alone, the fused score (v1, v2, k variants,
z-score), three leave-one-out ablations, and MMR variants. All settings are fixed in
`config.yaml` before labeling; nothing is tuned on the labels.

## Run (from the project root)

```bash
pip install sentence-transformers rank_bm25        # once
# 1. save the full CV as plain text: data/candidate/cv.txt (one paragraph per line)
python -m experiments.run sanity     # ~2 min, downloads 4 encoders
python -m experiments.run score      # ~10-20 min on Apple silicon (NLI on every listing skill)
python -m experiments.run pool       # writes experiments/out/pool/
# 2. two annotators label independently (see pool/LABELING_GUIDE.md):
#    open pool/labeling_A.html and pool/labeling_B.html in a browser, download the CSVs,
#    and save them over pool/labels_A.csv and pool/labels_B.csv
python -m experiments.run evaluate   # results.md, results.json (main table: annotator B)
python -m pytest tests/test_experiments.py
```

`evaluate --gold B|A|mean` picks the labels of the main table (default B; annotator A saw
the main system's top 10 before labeling). The other choices are written as additional
tables; Cohen's κ is always reported.

`--test` runs every stage with a hashing encoder and a lexical NLI stand-in (no downloads);
use it only to check the code path.

## Outputs (`experiments/out/`)

- `sanity.json`, `sanity_pairs.csv` – encoder check, selected encoder, thresholds
- `scores.csv` – every score per listing (new and reported), `skill_level_scores.csv`
- `rankings.json` – full ranking of every system
- `diagnostics.json` – spreads, length correlations, top-10 overlaps, ILD@10 (no labels needed)
- `pool/` – blind judging pool, labeling pages, guide; `pool_sources_…csv` must not be opened before labeling ends
- `results.md`, `results.json` – NDCG@10 and P@10 with 95% bootstrap intervals (resampling
  the judged listings; paired differences to the main system), Cohen's κ, per-score validity

The outputs contain the CV; they are excluded from git.
