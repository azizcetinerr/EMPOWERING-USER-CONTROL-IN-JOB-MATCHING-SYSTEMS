"""Additional analyses requested in review (Oct 2026). No relevance label of the volunteering snapshot is used.

  python -m experiments.extra_analyses postprocess   # XLM-R postprocessing on the ESCO pairs
  python -m experiments.extra_analyses profiles      # TalentCLEF CVs as profiles against the snapshot
                                                     # (also re-scores M_old on TalentCLEF with per-document batches)

postprocess
    The reported run's encoding (xlm-roberta-base, mean over all positions including padding, batches of 64
    distinct texts as in `run.py sanity_esco`) is evaluated on the 4,000 ESCO pairs after three label-free
    postprocessing steps whose statistics are estimated on the encoded ESCO phrases themselves:
    per-dimension standardization (Timkey & van Schijndel, 2021), all-but-the-top with D principal components
    removed after centering (Mu & Viswanath, 2018), and whitening (Su et al., 2021).
    Writes out/extra/postprocess.json.

profiles
    Each of the 472 TalentCLEF 2026 Task A CVs (English development set; skills from the SkillNER cache of
    `experiments.talentclef`) is used as a candidate profile against the 939 snapshot listings.
    For every profile the original matching score M_old (Eq. 1; listing skills as stored, encoded per listing
    in batches of 32 and CV skills per CV in batches of 32, as in the reported run) and the corrected M of v3
    (Eq. 2, unshrunk and shrunk with k = 5; M_meanmax as well) are computed over the listings, and the spread,
    the Spearman correlation with description length, and the overlap of first-ten lists between profiles are
    summarized over profiles. L needs a motivation letter and E an NLI pass over every CV paragraph; neither
    is computed here.
    The per-CV XLM-R vectors are reused to re-score M_old on TalentCLEF with per-document batches of 32
    (the stored M_old used batches of 64 alphabetically sorted distinct skills), and the four metrics and the
    exact sign-flip test of M - M_old are recomputed. Writes out/extra/profiles.json and
    out/extra/talentclef_mold_batching.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import core, metrics
from .anisotropy import ESCO_PAIRS, RawNoMaskXLMR
from .encoders import make_encoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "extra"
log = logging.getLogger("extra")
DIM741 = 741


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def auc_block(a, b, y, kind, n_boot, seed):
    cos = (unit(a) * unit(b)).sum(1)
    pos, neg = cos[y == 1], cos[y == 0]
    rng = np.random.default_rng(seed)
    boots = [metrics.roc_auc(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                             np.r_[np.ones(len(pos)), np.zeros(len(neg))]) for _ in range(n_boot)]
    out = {"auc_related_vs_unrelated": float(metrics.roc_auc(cos, y)),
           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "median_cosine_related": float(np.median(pos)), "median_cosine_unrelated": float(np.median(neg))}
    for r, u in itertools.product(["synonym", "broader"], ["easy", "hard"]):
        m = (kind == r) | (kind == f"unrelated_{u}")
        out[f"{r}_vs_{u}"] = float(metrics.roc_auc(cos[m], (kind[m] == r).astype(float)))
    return out


def postprocess(cfg, n_boot, seed):
    pairs = pd.read_csv(ESCO_PAIRS)
    kind = pairs["pair_type"].astype(str).to_numpy()
    enc = RawNoMaskXLMR(device=cfg["encoders"]["device"])
    cache: dict = {}
    a = enc.cached(pairs.skill_a.astype(str).tolist(), cache)
    b = enc.cached(pairs.skill_b.astype(str).tolist(), cache)
    y = pairs.related.to_numpy()
    X = np.stack(list(cache.values()))                      # every distinct ESCO phrase once
    mu, sd = X.mean(0), X.std(0)
    variants = {"raw": lambda Z: Z, "without_dim_741": lambda Z: np.delete(Z, DIM741, axis=1),
                "standardized": lambda Z: (Z - mu) / np.clip(sd, 1e-12, None)}
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    for D in (1, 2, 7):
        P = Vt[:D]
        variants[f"all_but_the_top_D{D}"] = lambda Z, P=P: (Z - mu) - (Z - mu) @ P.T @ P
    C = np.cov(Xc, rowvar=False)
    U, S, _ = np.linalg.svd(C)
    W = U / np.sqrt(np.clip(S, 1e-12, None))
    variants["whitening"] = lambda Z: (Z - mu) @ W
    variants["whitening_k256"] = lambda Z: (Z - mu) @ W[:, :256]
    # Cross-fitted variants: statistics estimated on the phrases of one half of the pairs and applied to
    # the other half (two folds, fixed seed), so that no pair is transformed with its own phrases.
    perm = np.random.default_rng(seed).permutation(len(pairs))
    folds = [perm[: len(perm) // 2], perm[len(perm) // 2:]]

    def fitted(Xf):
        m = Xf.mean(0)
        Xc_ = Xf - m
        _, _, Vt_ = np.linalg.svd(Xc_, full_matrices=False)
        U_, S_, _ = np.linalg.svd(np.cov(Xc_, rowvar=False))
        W_ = U_ / np.sqrt(np.clip(S_, 1e-12, None))
        s_ = Xf.std(0)
        return {"standardized": lambda Z: (Z - m) / np.clip(s_, 1e-12, None),
                "all_but_the_top_D7": lambda Z: (Z - m) - (Z - m) @ Vt_[:7].T @ Vt_[:7],
                "whitening_k256": lambda Z: (Z - m) @ W_[:, :256]}

    cross = {}
    for name in ("standardized", "all_but_the_top_D7", "whitening_k256"):
        cos = np.zeros(len(pairs))
        for k in range(2):
            fit_idx, ev_idx = folds[1 - k], folds[k]
            f = fitted(np.vstack([a[fit_idx], b[fit_idx]]))[name]
            cos[ev_idx] = (unit(f(a[ev_idx])) * unit(f(b[ev_idx]))).sum(1)
        pos, neg = cos[y == 1], cos[y == 0]
        rng = np.random.default_rng(seed)
        boots = [metrics.roc_auc(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                                 np.r_[np.ones(len(pos)), np.zeros(len(neg))]) for _ in range(n_boot)]
        cross[name] = {"auc_related_vs_unrelated": float(metrics.roc_auc(cos, y)),
                       "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                       "broader_vs_hard": float(metrics.roc_auc(
                           cos[(kind == "broader") | (kind == "unrelated_hard")],
                           (kind[(kind == "broader") | (kind == "unrelated_hard")] == "broader").astype(float)))}
        log.info(f"cross-fitted {name}: {cross[name]['auc_related_vs_unrelated']:.4f}")
    res = {"n_pairs": int(len(pairs)), "n_distinct_phrases": int(len(X)), "cross_fitted_two_folds": cross,
           "statistics_estimated_on": "the encoded ESCO phrases (no labels)", "variants": {}}
    for name, f in variants.items():
        res["variants"][name] = auc_block(f(a), f(b), y, kind, n_boot, seed)
        log.info(f"{name}: {res['variants'][name]['auc_related_vs_unrelated']:.4f}")
    return res


# ----------------------------------------------------------------------------- profiles
def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def top10(scores, ids):
    return set(ids[core.order(scores, ids)[:10]])


def per_doc_vectors(enc, lists):
    return {d: (enc.per_listing(l) if l else None) for d, l in lists.items()}


def mold_from_vectors(P, U):
    """Eq. 1 with unnormalized vectors (cosine)."""
    if P is None or U is None:
        return 0.0
    return 100 * float((unit(P) @ unit(U).T).max(1).mean())


def exact_sign_flip(d):
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def profiles(cfg, seed):
    from .talentclef import OUT as TC_OUT, load_qrels, query_metrics, METRICS
    df, _ = core.load_listings(cfg)
    ids = df["listing_id"].to_numpy()
    lists_v3 = df["skills"].tolist()
    has = np.array([len(l) > 0 for l in lists_v3])
    length = df["description"].str.len().to_numpy()
    sk = cfg["skills"]
    import ast
    raw = pd.read_csv(ROOT / cfg["paths"]["listings"], encoding="utf-8-sig").drop_duplicates("link")
    raw_lists = [core.clean_skills(ast.literal_eval(x) if isinstance(x, str) else [], set(sk["stopwords"]),
                                   sk["min_length"]) for x in raw["extracted_skills"]]
    assert len(raw_lists) == len(df)

    cv_sk = json.loads((TC_OUT / "cache" / "skills_cvs.json").read_text())
    q_sk = json.loads((TC_OUT / "cache" / "skills_queries.json").read_text())
    cvs = sorted(cv_sk)
    sanity = json.loads((ROOT / cfg["paths"]["out_dir"] / "sanity_esco.json").read_text())
    tl, th = sanity["tau_low"], sanity["tau_high"]
    idf = core.idf_weights(lists_v3)

    t0 = time.perf_counter()
    new = make_encoder(sanity["selected"], cfg["encoders"]["batch_size"], cfg["encoders"]["device"])
    lsk = sorted({s for l in lists_v3 for s in l})
    csk = sorted({s for c in cvs for s in cv_sk[c]})
    Lv, Cv = new.encode(lsk, "query"), new.encode(csk, "query")
    li, ci = {s: i for i, s in enumerate(lsk)}, {s: i for i, s in enumerate(csk)}
    log.info(f"new encoder done ({time.perf_counter() - t0:.0f}s)")

    old = RawNoMaskXLMR(device=cfg["encoders"]["device"])
    list_vecs = [old.per_listing(l) if l else None for l in raw_lists]
    log.info(f"listing XLM-R vectors done ({time.perf_counter() - t0:.0f}s)")
    cv_vecs = per_doc_vectors(old, {c: cv_sk[c] for c in cvs})
    q_vecs = per_doc_vectors(old, q_sk)
    log.info(f"CV/posting XLM-R vectors done ({time.perf_counter() - t0:.0f}s)")

    rows, tops = [], {"M_old": [], "M": [], "M_unshrunk": []}
    Lunit = [unit(v) if v is not None else None for v in list_vecs]
    for c in cvs:
        U = Cv[[ci[s] for s in cv_sk[c]]]
        best = dict(zip(lsk, (Lv @ U.T).max(1)))
        m_cov, m_mm = core.matching_scores(lists_v3, best, idf, tl, th)
        m_sh = core.shrink_to_median(m_cov, lists_v3, idf, cfg["skills"]["shrinkage_k"])
        Uo = unit(cv_vecs[c])
        m_old = np.array([100 * float((P @ Uo.T).max(1).mean()) if P is not None else 0.0 for P in Lunit])
        r = {"cv": c, "n_cv_skills": len(cv_sk[c])}
        for name, x in {"M_old": m_old, "M_unshrunk": m_cov, "M": m_sh, "M_meanmax": m_mm}.items():
            v = x[has]
            r[f"{name}_sd"], r[f"{name}_min"], r[f"{name}_max"] = float(v.std(ddof=1)), float(v.min()), float(v.max())
            r[f"{name}_rho_length"] = spearman(x, length)
        r["M~M_old"] = spearman(m_sh[has], m_old[has])
        rows.append(r)
        tops["M_old"].append(top10(m_old, ids)); tops["M"].append(top10(m_sh, ids))
        tops["M_unshrunk"].append(top10(m_cov, ids))
    R = pd.DataFrame(rows)
    q = lambda x: {"median": float(np.median(x)), "p05": float(np.percentile(x, 5)),
                   "p95": float(np.percentile(x, 95)), "min": float(np.min(x)), "max": float(np.max(x))}
    rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(len(cvs), 2, replace=False)) for _ in range(2000)]
    summary = {"n_profiles": len(cvs), "n_listings": int(len(df)), "n_listings_with_skills": int(has.sum()),
               "cv_skills_median": float(np.median([len(cv_sk[c]) for c in cvs])),
               "tau_low": tl, "tau_high": th, "shrinkage_k": cfg["skills"]["shrinkage_k"],
               "per_profile": {k: q(R[k]) for k in R.columns if k not in ("cv",)},
               "share_profiles_M_old_sd_below_0.1": float((R["M_old_sd"] < 0.1).mean()),
               "share_profiles_M_old_range_within_1_point": float(((R["M_old_max"] - R["M_old_min"]) < 1).mean()),
               "top10_overlap_between_random_profile_pairs": {
                   k: q([len(v[i] & v[j]) for i, j in pairs]) for k, v in tops.items()},
               "listings_in_top10_of_any_profile": {k: len(set().union(*v)) for k, v in tops.items()}}

    # ---- M_old on TalentCLEF with per-document batches
    qrels = load_qrels()
    stored = pd.read_csv(TC_OUT / "scores.csv", dtype={"qid": str, "doc": str})
    perq = pd.read_csv(TC_OUT / "per_query.csv", dtype={"qid": str})
    perq = perq[perq["setting"] == "full_corpus"]
    docs = np.array(sorted(stored["doc"].unique()))
    tc_rows, mold_rows = [], []
    for qid in sorted(q_sk):
        P = q_vecs[qid]
        s_new = np.array([mold_from_vectors(P, cv_vecs.get(d)) for d in docs])
        g = stored[stored["qid"] == qid].set_index("doc").loc[docs]
        mold_rows.append({"qid": qid, "spearman_with_stored_M_old": spearman(s_new, g["M_old"].to_numpy()),
                          "max_abs_diff": float(np.abs(s_new - g["M_old"].to_numpy()).max())})
        rank = docs[core.order(s_new, docs)].tolist()
        tc_rows.append({"qid": qid, **query_metrics(rank, qrels.get(qid, set()))})
    T = pd.DataFrame(tc_rows).set_index("qid")
    Mq = perq[perq["system"] == "M"].set_index("qid")
    Oq = perq[perq["system"] == "M_old"].set_index("qid")
    tc = {"batching": "per document, consecutive batches of 32 skills (as the reported run's vectorizer)",
          "stored_batching": "distinct skills sorted alphabetically, batches of 64",
          "agreement_with_stored": {"spearman_median": float(np.median([r["spearman_with_stored_M_old"] for r in mold_rows])),
                                    "spearman_min": float(np.min([r["spearman_with_stored_M_old"] for r in mold_rows])),
                                    "max_abs_diff": float(np.max([r["max_abs_diff"] for r in mold_rows]))},
          "means_per_document_batching": T[METRICS].mean().to_dict(),
          "means_stored": Oq[METRICS].mean().to_dict(), "comparisons_M_minus_M_old": {}}
    for m in METRICS:
        d = (Mq[m] - T[m]).loc[sorted(q_sk)].to_numpy()
        tc["comparisons_M_minus_M_old"][m] = {"mean_diff": float(d.mean()), "p_exact": exact_sign_flip(d),
                                              "positive": int((d > 0).sum()), "zero": int((d == 0).sum())}
    return summary, R, tc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["postprocess", "profiles"])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config_v3.yaml")))
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    env = {"python": platform.python_version(), "platform": platform.platform(), "seed": cfg["seed"]}
    if a.stage == "postprocess":
        res = postprocess(cfg, a.n_bootstrap, cfg["seed"])
        res["environment"] = env
        (OUT / "postprocess.json").write_text(json.dumps(res, indent=2))
    else:
        summary, R, tc = profiles(cfg, cfg["seed"])
        summary["environment"] = tc["environment"] = env
        (OUT / "profiles.json").write_text(json.dumps(summary, indent=2))
        R.to_csv(OUT / "profiles_per_cv.csv", index=False)
        (OUT / "talentclef_mold_batching.json").write_text(json.dumps(tc, indent=2))
    log.info(f"written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
