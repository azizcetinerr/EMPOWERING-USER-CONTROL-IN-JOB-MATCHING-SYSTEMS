# Corrected-system experiments

Re-scores the December 2025 snapshot (952 records, 939 distinct listings) with the corrected ranker,
compares it with baselines and with the reported run on label-free diagnostics, checks the encoder on
skill pairs, and validates the matching scores on an external labeled dataset (TalentCLEF 2026 Task A).

**No human relevance labels were collected in this study.** The labeling tools (`run.py pool`,
`run.py evaluate`, `labeling_template.html`) are still in the repository, but they were not used:
the judging pools under `out/*/pool/` are empty templates, and no NDCG or precision value exists for
the volunteering listings. The only labeled evaluation is the TalentCLEF one below.

## What changed relative to the reported run

| Component | Reported run | Corrected |
|---|---|---|
| Encoder | XLM-R, mean over all positions incl. padding | multilingual MPNet (`paraphrase-multilingual-mpnet-base-v2`), chosen among XLM-R (masked / unmasked), multilingual MPNet, LaBSE and multilingual-E5 by ROC-AUC on skill pairs (see Versions) |
| M | mean of best cosine per listing skill | IDF-weighted soft coverage; thresholds tau = median cosine of unrelated / related pairs |
| S → L | AI-detector constant + letter–listing cosine | letter–listing cosine only (detector removed); called *letter–listing relevance* |
| R → E | logistic of the entailment logit, first 10 CV lines, host bonuses | softmax entailment probability, all CV paragraphs, IDF-weighted; host terms removed (fields are empty); called *CV evidence* |
| Fusion | z-score 50 + 20z, clipped | percentile rank (main), z-score reported as a variant; weights unchanged (0.5/0.3/0.2) |
| Keyword bonus | substring test, 909/952 listings | removed |
| Duplicates | 13 duplicated records | removed by link |
| Listing skills | SkillNER output as stored (on non-English listings mostly function words: "los", "en", "ses") | skills that are a stopword in any of 24 spaCy languages removed; non-English listings keep only skills also extracted from an English listing; listings left without skills get the median M and E |
| Few-skill listings | M and E are means over 1–3 skills for ~10% of listings, so they take extreme values | M and E shrunk toward the median: (n·score + 5·median)/(n + 5), n = IDF-weighted skill count; unshrunk scores kept |
| Long descriptions | – (the selected encoder reads 128 tokens; 99% of descriptions are longer, median 489) | 120-token chunks with 20-token overlap: L and dense = best chunk, MMR/ILD = normalized chunk mean; the dense baseline's profile query (CV + letter) is chunked too; first-128-token scores kept as `L_trunc128`, `dense_trunc128` |
| MMR | λ = 0.2 on mean skill vectors | off and λ ∈ {0.9, 0.7, 0.5, 0.2} on description embeddings, all reported |

## Versions of the fused ranker

All three are scored as separate systems; v1 and v2 are kept as version history.

- **v1** – `config_v1.yaml`: the configuration fixed before any ranking was inspected (stored skills,
  no shrinkage, listings without skills get 0, L on the first 128 tokens). In the outputs its label is
  `Fused (v1, pre-registered)`; this is a historical label, there is no external registration.
- **v2** – `config.yaml`: adds the language filter, the shrinkage (k = 5; k = 0, 2.5 and 10 as variants)
  and the chunking. These were added after inspecting the top 10 of v1 and of intermediate versions;
  no labels were involved. Encoder and tau come from 100 hand-written skill pairs (`skill_pairs.csv`,
  drafted with an AI assistant and checked by the authors).
- **v3, the main system** – `config_v3.yaml`: v2 with the encoder and tau taken from public ESCO
  skill relations instead of the hand-written pairs. `python -m experiments.esco_pairs` builds
  `esco_pairs.csv` from ESCO v1.2.1 (1,000 pairs each of synonym, broader, easy-unrelated and
  hard-unrelated, 4,000 in total; every concept and label used once), and
  `python -m experiments.run sanity_esco` writes `out/sanity_esco.json`. The ESCO check selected the
  same encoder as v2; only tau changed (0.106 / 0.809 → 0.165 / 0.783). MMR variants and the
  diagnostics refer to v3.

Code and configurations are frozen by the git tags `pre-labeling` (v1, v2) and `esco-talentclef`
(v3, TalentCLEF). The tag name `pre-labeling` predates the decision not to collect labels.

## Run (from the project root)

```bash
pip install sentence-transformers rank_bm25 spacy scikit-learn   # once
# save the full CV as plain text: data/candidate/cv.txt (one paragraph per line)
python -m experiments.run sanity        # encoder check on the 100 hand-written pairs (v2)
python -m experiments.esco_pairs        # ESCO pairs (needs the ESCO v1.2.1 CSV files)
python -m experiments.run sanity_esco   # encoder check on the ESCO pairs (v3)
python -m experiments.run score         # ~10-20 min on Apple silicon (NLI on every listing skill)
python -m experiments.run diagnostics   # rebuild rankings/diagnostics from stored scores, no CV needed
python -m experiments.anisotropy        # outlier dimension of the reported run's XLM-R vectors
python -m experiments.talentclef --dataset-version <version>   # external validation (below)
```

`run.py diagnostics` re-encodes only the listing descriptions (for MMR and ILD) and checks that the
stored v1, v2 and v3 rankings are reproduced from `scores.csv`; the checks are written to
`diagnostics.json` under `checks`. `--test` runs a stage with a hashing encoder and a lexical NLI
stand-in (no downloads); use it only to check the code path.

`experiments/anisotropy.py` reads the skill vectors that the reported run stored in
`data/raw/youth_portal_projeler_detayli.csv`. That column holds numpy's printed form, which is
abbreviated with "..." for listings with more than one skill, so complete vectors exist only for
single-skill listings; the script therefore also re-encodes 2,000 listing skills exactly as the
reported run did and checks the result against the printed values.

## External validation: TalentCLEF 2026 Task A

`experiments/talentclef.py` scores the English development set of TalentCLEF 2026 Task A
(Zenodo, doi:10.5281/zenodo.17625261, CC BY 4.0; place it under
`data/external/talentclef/TaskA/development/en/`). The job posting is the query and the CV is the
document; every query ranks all 472 CVs (main table), and as an additional table its candidate set
(labeled CVs + BM25 top 100). Systems: BM25, Dense, M_old (the reported run's encoder and formula),
M_meanmax, M, E and M+E. Encoder and tau come from `sanity_esco.json`; nothing is tuned on
TalentCLEF. Paired randomization tests (10,000 sign flips), bootstrap 95% intervals and Holm
correction over the three comparisons of M per metric (vs BM25, Dense, M_old).

## Outputs (`experiments/out/`)

Tracked in git (no CV text; counts, listing identifiers and scores only):
- `diagnostics.json` – spreads, length correlations, score correlations, top-10 overlaps, ILD@10,
  version checks (no labels needed)
- `scores.csv` – every score per listing (new and reported)
- `sanity_esco.json` – ESCO encoder check, selected encoder, tau
- `anisotropy.json` – outlier-dimension statistics of the reported run's vectors
- `talentclef/results.json`, `results.md`, `per_query.csv` – external validation

Not released (they contain the CV or evidence derived from it): `data/candidate/cv.txt`,
`cv_skills_used.txt`, `skill_level_scores.csv` (per-skill CV similarity and entailment), `pool/`.
Other outputs (`sanity.json`, `sanity_pairs.csv`, `rankings.json`) are written locally.
