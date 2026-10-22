"""Revision statistics for the TalentCLEF comparison (reviewer requests, Oct 2026).

  python -m experiments.extra_stats

Reads out/talentclef/per_query.csv (full corpus) and writes out/talentclef/extra_stats.json:
  * exact two-sided sign-flip test over all 2^n sign patterns (n = 10 queries -> 1,024 patterns),
    which replaces the 10,000 random sign flips of talentclef.paired_randomization;
  * Holm adjustment within each metric (3 planned comparisons, as in the paper) and over all
    12 comparisons (4 metrics x 3) as one family, for both the exact and the stored p-values.
No score is recomputed; only the per-query metrics stored by `python -m experiments.talentclef` are used.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.talentclef import COMPARISONS, METRICS, holm

OUT = Path(__file__).resolve().parent / "out" / "talentclef"


def exact_sign_flip(d) -> float:
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


def main():
    P = pd.read_csv(OUT / "per_query.csv")
    P = P[P["setting"] == "full_corpus"]
    stored = json.loads((OUT / "results.json").read_text())
    stored = stored.get("settings", stored)["full_corpus"]["comparisons"]
    exact, old = {}, {}
    for m in METRICS:
        piv = P.pivot(index="qid", columns="system", values=m)
        for a, b in COMPARISONS:
            key = f"{m} | {a} vs {b}"
            exact[key] = exact_sign_flip(piv[a] - piv[b])
            old[key] = stored[m][f"{a} vs {b}"]["p"]
    res = {"n_queries": int(P["qid"].nunique()), "n_sign_patterns": 2 ** int(P["qid"].nunique()),
           "comparisons": {}}
    for m in METRICS:
        keys = [k for k in exact if k.startswith(m + " |")]
        w_exact = holm({k: exact[k] for k in keys})
        w_old = holm({k: old[k] for k in keys})
        for k in keys:
            res["comparisons"][k] = {"p_random_10000": old[k], "p_exact": exact[k],
                                     "p_holm_within_metric_random": w_old[k],
                                     "p_holm_within_metric_exact": w_exact[k]}
    a_exact, a_old = holm(exact), holm(old)
    for k in exact:
        res["comparisons"][k]["p_holm_12_random"] = a_old[k]
        res["comparisons"][k]["p_holm_12_exact"] = a_exact[k]
    (OUT / "extra_stats.json").write_text(json.dumps(res, indent=2))
    for k, v in res["comparisons"].items():
        print(f"{k:28s} " + "  ".join(f"{n}={x:.4f}" for n, x in v.items()))


if __name__ == "__main__":
    main()
