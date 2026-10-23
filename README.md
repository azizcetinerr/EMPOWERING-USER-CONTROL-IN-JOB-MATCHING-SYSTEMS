# Label-free checks for component-score failure: code and outputs

This repository contains the code and outputs behind the manuscript

> A. Kayabaş, A. M. Çetiner. *Label-Free Checks for Silent Component-Score Failure in Multi-Score Rankers: A Volunteering Case Study.* Submitted to *Expert Systems*, 2026.

The study audits a three-score content-based ranker for European Youth Portal volunteering listings
(one candidate profile, a snapshot of 939 listings from December 2025), diagnoses why its scores did not
separate listings, corrects the components, and evaluates the matching score on TalentCLEF 2026 Task A.
Every value reported in the manuscript is produced by the scripts below from files in this repository,
except where a step needs the candidate CV or the scraped listing file (see *What is not released*).

The release is frozen by the tag **`expert-systems-2026`**. The configurations of the corrected ranker were
fixed earlier and are frozen by `pre-labeling` (v1, v2; commit `82e7912`) and `esco-talentclef`
(v3 and the TalentCLEF comparison; commit `03ab459`).

**No human relevance labels and no data from other users were collected.** The ranker was run only
locally for one author's profile. A web interface for a planned multi-user study is kept on the branch
`web-interface`; it has not been deployed and is not part of this study.

## Installation

```bash
pip install -r requirements.txt
pip install sentence-transformers rank_bm25 scikit-learn
python -m spacy download en_core_web_lg
```

Run every command from the repository root.

## Which script produces which result

| Manuscript | Command | Output |
|---|---|---|
| Original run: score distributions, D2 (detector output, context term), D4 (keyword bonus), D6 (score correlations) | `python PAPER/verify_numbers.py outputs/results/IP5_S_recomputed.csv` | printed |
| D6: mean pairwise cosine 0.9972 of the original MMR step | (log of the original run) | `experiments/out/original_run_log_excerpt.txt` |
| D1: dominant coordinate 741, re-encoding check, Appendix A | `python -m experiments.anisotropy` | `experiments/out/anisotropy.json` |
| Table 4: encoder check on the 100 hand-written pairs | `python -m experiments.run sanity` | `experiments/out/sanity.json` |
| Table 4: ESCO pairs and encoder check | `python -m experiments.esco_pairs` then `python -m experiments.run sanity_esco` | `experiments/esco_pairs.csv`, `experiments/out/sanity_esco.json` |
| Table 4: XLM-R without dimension 741 | `python -m experiments.dim741_pair_auc` | `experiments/out/dim741_pair_auc.json` |
| Table 4: postprocessed XLM-R rows | `python -m experiments.extra_analyses postprocess` | `experiments/out/extra/postprocess.json` |
| Corrected scores M, L, E of v1–v3 | `python -m experiments.run score` | `experiments/out/scores.csv` |
| Tables 3 and 5, Figure 3, Section 7.2 | `python -m experiments.run diagnostics`, `python -m experiments.plot_mmr` | `experiments/out/diagnostics.json`, `experiments/out/rankings.json` |
| Section 7.2 (other profiles), Section 7.3 (batching of M_old) | `python -m experiments.extra_analyses profiles` | `experiments/out/extra/profiles.json`, `profiles_per_cv.csv`, `talentclef_mold_batching.json` |
| Table 6 (means), Table A1 | `python -m experiments.talentclef` | `experiments/out/talentclef/results.json`, `per_query.csv` |
| Table 6 (exact tests, Holm over 3 and 12 comparisons) | `python -m experiments.extra_stats` | `experiments/out/talentclef/extra_stats.json` |
| Section 8.3: IDF weights | `python -m experiments.idf_stats` | `experiments/out/idf_stats.json` |
| Sections 1, 3.2, 9: listings still open on 4 Feb and 18 Oct 2026 | `python experiments/snapshot_compare/compare.py` | `experiments/snapshot_compare/compare.json` |

`experiments/README.md` describes the corrected ranker, its three versions and the outputs in more detail.
TalentCLEF 2026 Task A (version 0.1.0, Zenodo doi:10.5281/zenodo.18449283, CC BY 4.0) and ESCO v1.2.1 must be
downloaded separately; the expected locations are given there.

## What is not released

- The candidate CV (`data/candidate/cv.txt`) and outputs derived from it (`cv_skills_used.txt`,
  `skill_level_scores.csv`, `pool/`): personal data. `run score` and `extra_analyses profiles` need the CV;
  `run diagnostics` and the TalentCLEF scripts do not.
- The scraped listing file `data/raw/youth_portal_projeler_detayli.csv` read by `experiments.anisotropy`;
  the listings themselves are public on the European Youth Portal.

## Repository layout

```
config/        configuration of the original ranker (includes the motivation letter of the reported run)
src/           original ranker (skill extraction, scoring, pipeline)
experiments/   corrected ranker, diagnostics, encoder checks, TalentCLEF evaluation, additional analyses
outputs/       score files of the original run used by PAPER/verify_numbers.py
PAPER/         scripts that recompute the values of the original run
web/           local web interface of the original ranker (run only locally)
docs/          development notes of the original system
```

## License

The code is released under the [MIT License](LICENSE).
