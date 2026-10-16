"""Ranking metrics for one candidate with graded labels on a judged pool.

Unjudged listings are removed from each ranking before the metrics are computed
(condensed lists; Sakai, 2007). Gains are 2^g - 1 for a grade g in [0, 2]; with two
annotators g is their mean grade. A listing counts as relevant for P@k when g >= 1.
"""
from __future__ import annotations

import numpy as np


def dcg(gains, k):
    g = np.asarray(gains, dtype=float)[:k]
    return float(((2 ** g - 1) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(ranked_grades, all_grades, k):
    ideal = dcg(sorted(all_grades, reverse=True), k)
    return dcg(ranked_grades, k) / ideal if ideal > 0 else 0.0


def precision_at_k(ranked_grades, k, threshold=1.0):
    g = np.asarray(ranked_grades, dtype=float)[:k]
    return float((g >= threshold).sum() / k)


def condensed(ranking, judged: dict):
    """Grades of the judged listings in ranking order."""
    return [judged[i] for i in ranking if i in judged]


def evaluate_rankings(rankings: dict, judged: dict, k: int) -> dict:
    allg = list(judged.values())
    out = {}
    for name, r in rankings.items():
        g = condensed(r, judged)
        out[name] = {"ndcg": ndcg_at_k(g, allg, k), "p": precision_at_k(g, k)}
    return out


def bootstrap(rankings: dict, judged: dict, k: int, n_boot: int, seed: int,
              reference: str | None = None) -> dict:
    """Percentile 95% intervals from resampling the judged listings with replacement.
    The same resamples are used for every system, so differences to `reference` are paired."""
    rng = np.random.default_rng(seed)
    ids = np.array(list(judged.keys()))
    pos = {name: {i: p for p, i in enumerate(r)} for name, r in rankings.items()}
    res = {name: {"ndcg": [], "p": []} for name in rankings}
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        grades = np.array([judged[i] for i in sample])
        for name in rankings:
            p = np.array([pos[name][i] for i in sample])
            g = grades[np.argsort(p, kind="stable")]
            res[name]["ndcg"].append(ndcg_at_k(g, grades, k))
            res[name]["p"].append(precision_at_k(g, k))
    out = {}
    for name, d in res.items():
        out[name] = {}
        for m, vals in d.items():
            v = np.array(vals)
            out[name][f"{m}_ci"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if reference and name != reference:
                diff = v - np.array(res[reference][m])
                out[name][f"{m}_diff_vs_{reference}"] = (
                    float(diff.mean()), float(np.percentile(diff, 2.5)),
                    float(np.percentile(diff, 97.5)), float((diff <= 0).mean()))
    return out


def intra_list_diversity(ranking, emb, k):
    """1 - mean pairwise cosine of the first k items (Ziegler et al., 2005)."""
    top = np.asarray(ranking)[:k]
    S = emb[top] @ emb[top].T
    iu = np.triu_indices(len(top), 1)
    return float(1 - S[iu].mean())


def roc_auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))
