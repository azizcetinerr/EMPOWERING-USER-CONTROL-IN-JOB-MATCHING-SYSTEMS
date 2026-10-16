"""Experiments for the corrected ranker on the December 2025 snapshot.

Stages (run from the project root):
  python -m experiments.run sanity     # encoder check on labeled skill pairs, picks the encoder
  python -m experiments.run score      # corrected scores, baselines, all rankings, diagnostics
  python -m experiments.run pool       # blind judging pool + labeling pages for two annotators
  python -m experiments.run evaluate   # NDCG@10, P@10, bootstrap CIs, kappa, score validity
  python -m experiments.run all        # sanity + score + pool

Options: --config PATH, --test (hashing encoder and lexical NLI; no downloads, for smoke tests)
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from . import core, metrics
from .encoders import NLIScorer, make_encoder

log = logging.getLogger("experiments")
ROOT = core.ROOT


def _out(cfg) -> Path:
    p = ROOT / cfg["paths"]["out_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _encoder_specs(cfg, test):
    if test:
        return [{"name": "hashing-a", "backend": "hashing"}, {"name": "hashing-b", "backend": "hashing"}]
    return cfg["encoders"]["candidates"]


# ----------------------------------------------------------------------------- sanity
def stage_sanity(cfg, test=False):
    out = _out(cfg)
    pairs = pd.read_csv(ROOT / cfg["paths"]["skill_pairs"])
    listings, _ = core.load_listings(cfg)
    rng = np.random.default_rng(cfg["seed"])
    pool = sorted({s for lst in listings["skills"] for s in lst})
    rand = [(pool[i], pool[j]) for i, j in rng.integers(0, len(pool), size=(2000, 2)) if i != j]

    rows, sims = [], {}
    for spec in _encoder_specs(cfg, test):
        enc = make_encoder(spec, cfg["encoders"]["batch_size"], cfg["encoders"]["device"])
        a = enc.encode(pairs.skill_a.tolist(), "query")
        b = enc.encode(pairs.skill_b.tolist(), "query")
        cos = (a * b).sum(1)
        ra = enc.encode([x for x, _ in rand], "query")
        rb = enc.encode([y for _, y in rand], "query")
        rcos = (ra * rb).sum(1)
        pos, neg = cos[pairs.related == 1], cos[pairs.related == 0]
        rows.append({"encoder": enc.label, "name": spec["name"], "backend": spec["backend"],
                     "auc": metrics.roc_auc(cos, pairs.related),
                     "related_median": float(np.median(pos)), "unrelated_median": float(np.median(neg)),
                     "gap": float(np.median(pos) - np.median(neg)),
                     "random_listing_pairs_median": float(np.median(rcos)),
                     "random_listing_pairs_p05_p95": [float(np.percentile(rcos, 5)),
                                                      float(np.percentile(rcos, 95))]})
        sims[enc.label] = cos
        log.info(f"{enc.label}: AUC={rows[-1]['auc']:.3f} related={rows[-1]['related_median']:.3f} "
                 f"unrelated={rows[-1]['unrelated_median']:.3f} random={rows[-1]['random_listing_pairs_median']:.3f}")

    table = pd.DataFrame(rows)
    best = table.sort_values(["auc", "gap"], ascending=False, kind="stable").iloc[0]
    spec = next(s for s in _encoder_specs(cfg, test)
                if s["name"] == best["name"] and s["backend"] == best["backend"])
    result = {"selected": spec, "selection_rule": "highest ROC-AUC on the skill pairs (ties: larger gap)",
              "tau_low": best["unrelated_median"], "tau_high": best["related_median"],
              "encoders": rows, "n_pairs": int(len(pairs)), "n_related": int(pairs.related.sum())}
    _dump(result, out / "sanity.json")
    pd.concat([pairs, pd.DataFrame(sims)], axis=1).to_csv(out / "sanity_pairs.csv", index=False)
    log.info(f"Selected encoder: {best['encoder']} (tau_low={best['unrelated_median']:.3f}, "
             f"tau_high={best['related_median']:.3f})")
    return result


# ----------------------------------------------------------------------------- score
def stage_score(cfg, test=False):
    out = _out(cfg)
    sanity = json.loads((out / "sanity.json").read_text())
    df, data_info = core.load_listings(cfg)
    cand = core.load_candidate(cfg)
    cv_skills = (core.clean_skills(cand["cv_text"].replace(",", "\n").splitlines(),
                                   set(cfg["skills"]["stopwords"]), cfg["skills"]["min_length"])
                 if test else core.extract_cv_skills(cfg, cand["cv_text"]))
    (out / "cv_skills_used.txt").write_text("\n".join(cv_skills), encoding="utf-8")
    log.info(f"{len(df)} listings, {len(cv_skills)} CV skills, {len(cand['paragraphs'])} CV paragraphs")

    # v1 (pre-registered): skill processing, fusion and chunking from config_v1.yaml
    cfg1 = yaml.safe_load((ROOT / cfg["paths"]["config_v1"]).read_text(encoding="utf-8"))
    df1, _ = core.load_listings({**cfg1, "paths": cfg["paths"]})
    if not df1["listing_id"].equals(df["listing_id"]):
        raise ValueError("v1 and v2 listings are not aligned")
    df["skills_v1"] = df1["skills"]

    t0 = time.perf_counter()
    enc = make_encoder(sanity["selected"], cfg["encoders"]["batch_size"], cfg["encoders"]["device"])
    use_idf = cfg["matching"]["idf_weighting"]
    k_shrink = cfg["skills"]["shrinkage_k"]

    # Skill similarity and NLI evidence for every skill of v1 and v2
    best_sim = core.best_skill_similarity(enc, list(df["skills_v1"]) + list(df["skills"]), cv_skills)
    nli = NLIScorer(cfg["nli"]["model"], cfg["nli"]["batch_size"], cfg["encoders"]["device"],
                    backend="lexical" if test else "hf")
    evidence = core.skill_evidence(nli, list(best_sim), cand["paragraphs"], cfg["nli"]["hypothesis"])

    def skill_scores(lists, k):
        """M (soft IDF-weighted coverage), M_meanmax, E (CV evidence); k=None: no shrinkage, empty -> 0."""
        idf_ = core.idf_weights(lists)
        m, mm = core.matching_scores(lists, best_sim, idf_, sanity["tau_low"], sanity["tau_high"], use_idf)
        e = core.evidence_scores(lists, evidence, idf_, use_idf)
        if k is not None:
            m, e = (core.shrink_to_median(x, lists, idf_, k, use_idf) for x in (m, e))
        return m, mm, e, idf_

    df["M_unshrunk"], df["M_meanmax"], df["E_unshrunk"], idf = skill_scores(df["skills"], None)
    df["M"], _, df["E"], _ = skill_scores(df["skills"], k_shrink)
    df["M_v1"], _, df["E_v1"], _ = skill_scores(df["skills_v1"], cfg1["skills"].get("shrinkage_k"))
    k_variants = {f"k={k:g}": k for k in (0, k_shrink / 2, 2 * k_shrink)}
    for name, k in k_variants.items():
        df[f"M_{name}"], _, df[f"E_{name}"], _ = skill_scores(df["skills"], k)
    has = df["skills"].str.len() > 0
    pd.DataFrame({"skill": list(evidence), "evidence": list(evidence.values()),
                  "best_cv_similarity": [best_sim[s] for s in evidence],
                  "idf_v2": [idf.get(s) for s in evidence]}).sort_values("idf_v2").to_csv(
        out / "skill_level_scores.csv", index=False)

    # Descriptions longer than the encoder's 128-token window are encoded in overlapping chunks:
    # relevance = best chunk, document embedding (MMR, ILD) = normalized chunk mean.
    ch = cfg["encoders"]["passage_chunks"]
    chunks = enc.encode_chunked(df["description"].tolist(), "passage", ch["tokens"], ch["overlap"])
    D = core.mean_chunk_embedding(chunks)
    D_trunc = enc.encode(df["description"].tolist(), "passage")
    q_letter = enc.encode([cand["letter"]], "query")
    df["L"] = 100 * core.max_chunk_cosine(q_letter, chunks)
    df["L_trunc128"] = 100 * (D_trunc @ q_letter.T).ravel()
    ch1 = cfg1["encoders"].get("passage_chunks")
    df["L_v1"] = df["L_trunc128"] if not ch1 else 100 * core.max_chunk_cosine(
        q_letter, chunks if ch1 == ch else
        enc.encode_chunked(df["description"].tolist(), "passage", ch1["tokens"], ch1["overlap"]))

    # Baselines on the whole profile (CV text + letter) against the description
    query = cand["cv_text"] + "\n" + cand["letter"]
    df["bm25"] = core.bm25_scores(query, df["description"])
    df["tfidf"] = core.tfidf_scores(query, df["description"])
    # The profile (~1,500 tokens) is chunked too: profile vector = normalized mean of its chunks
    qv = core.mean_chunk_embedding(enc.encode_chunked([query], "query", ch["tokens"], ch["overlap"]))
    df["dense_profile"] = core.max_chunk_cosine(qv, chunks)
    df["dense_trunc128"] = (D_trunc @ enc.encode([query], "query").T).ravel()
    elapsed = time.perf_counter() - t0

    # Fused variants (weights fixed in the config; never tuned on labels)
    w = cfg["fusion"]["weights"]
    sc = {"M": df["M"], "L": df["L"], "E": df["E"]}
    main_norm = cfg["fusion"]["normalization"]
    df["F_rank"] = core.fuse(sc, w, "rank")
    df["F_zscore"] = core.fuse(sc, w, "zscore")
    df["F_noL"] = core.fuse(sc, {k: v for k, v in w.items() if k != "L"}, main_norm)
    df["F_noE"] = core.fuse(sc, {k: v for k, v in w.items() if k != "E"}, main_norm)
    df["F_noM"] = core.fuse(sc, {k: v for k, v in w.items() if k != "M"}, main_norm)
    df["F_v1"] = core.fuse({"M": df["M_v1"], "L": df["L_v1"], "E": df["E_v1"]},
                           cfg1["fusion"]["weights"], cfg1["fusion"]["normalization"])
    for name in k_variants:
        df[f"F_{name}"] = core.fuse({"M": df[f"M_{name}"], "L": df["L"], "E": df[f"E_{name}"]}, w, main_norm)
    main = "F_rank" if main_norm == "rank" else "F_zscore"

    ids = df["listing_id"].to_numpy()
    rk = lambda col: ids[core.order(df[col])].tolist()
    main_label = "Fused (v2)" if main == "F_rank" else "Fused (v2, z-score)"
    rankings = {
        "original_A (reported run)": rk("Fair_Score"),
        "original_M (reported run)": rk("M_Score_Raw"),
        "BM25 (profile)": rk("bm25"),
        "TF-IDF (profile)": rk("tfidf"),
        "Dense (profile)": rk("dense_profile"),
        "M only": rk("M"),
        "L only": rk("L"),
        "E only": rk("E"),
        "Fused (v2)": rk("F_rank"),
        "Fused (v1, pre-registered)": rk("F_v1"),
        **{f"Fused (v2), {name}": rk(f"F_{name}") for name in k_variants},
        "Fused (v2, z-score)": rk("F_zscore"),
        "Fused (v2) without L": rk("F_noL"),
        "Fused (v2) without E": rk("F_noE"),
        "Fused (v2) without M": rk("F_noM"),
    }
    for lam in cfg["mmr"]["lambdas"]:
        rankings[f"{main_label} + MMR λ={lam}"] = ids[core.mmr_order(df[main], D, cfg["mmr"]["k"], lam)].tolist()

    # Diagnostics that do not need labels
    k = cfg["evaluation"]["k"]
    pos_of = {i: p for p, i in enumerate(ids)}
    ild = {n: metrics.intra_list_diversity([pos_of[i] for i in r], D, k) for n, r in rankings.items()}
    length = df["description"].str.len()
    spread = lambda x: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
                        "min": float(np.min(x)), "max": float(np.max(x))}
    sims = np.array(list(best_sim.values()))
    diag = {
        "data": data_info,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "encoder": enc.label, "nli": cfg["nli"]["model"], "scoring_seconds": elapsed},
        "cv": {"skills": len(cv_skills), "paragraphs": len(cand["paragraphs"])},
        "tau": {"low": sanity["tau_low"], "high": sanity["tau_high"]},
        "distinct_listing_skills": len(best_sim),
        "share_listing_skills_at_or_above_tau_high": float((sims >= sanity["tau_high"]).mean()),
        "share_listing_skills_at_or_below_tau_low": float((sims <= sanity["tau_low"]).mean()),
        "spread_over_listings_with_skills": {
            "M": spread(df.loc[has, "M"]), "M_unshrunk": spread(df.loc[has, "M_unshrunk"]),
            "E_unshrunk": spread(df.loc[has, "E_unshrunk"]), "M_meanmax": spread(df.loc[has, "M_meanmax"]),
            "original_M_raw": spread(df.loc[has, "M_Score_Raw"]),
            "L": spread(df["L"]), "original_S_context": spread(df["Context_Score"]),
            "E": spread(df.loc[has, "E"]), "original_R_raw": spread(df.loc[has, "R_Score_Raw"])},
        "passage_chunks": {"tokens": ch["tokens"], "overlap": ch["overlap"],
                           "share_descriptions_with_more_than_one_chunk": float(np.mean([len(c) > 1 for c in chunks])),
                           "median_chunks": float(np.median([len(c) for c in chunks])),
                           "max_chunks": int(max(len(c) for c in chunks))},
        "shrinkage_k": {"main": k_shrink, "variants": list(k_variants.values())},
        "spearman_with_description_length": {
            c: float(spearmanr(df[c], length).correlation)
            for c in ["L", "L_trunc128", "dense_profile", "dense_trunc128", "Context_Score"]},
        "spearman_between_scores": {
            f"{a}~{b}": float(spearmanr(df[a], df[b]).correlation)
            for a, b in [("M", "L"), ("M", "E"), ("L", "E"), ("M", "M_Score_Raw"), ("L", "L_trunc128"),
                         ("dense_profile", "dense_trunc128"), ("F_rank", "F_v1"),
                         ("F_rank", "F_zscore"), ("F_rank", "Fair_Score"), ("M", "bm25")]},
        "top10_overlap_v1_v2": len(set(rankings["Fused (v1, pre-registered)"][:k])
                                   & set(rankings["Fused (v2)"][:k])),
        "top10_overlap_with_main": {n: len(set(r[:k]) & set(rankings[main_label][:k]))
                                    for n, r in rankings.items()},
        f"ILD@{k}": ild,
    }
    _dump(diag, out / "diagnostics.json")
    _dump({"main": main_label, "rankings": rankings}, out / "rankings.json")
    keep = ["listing_id", "link", "M", "M_unshrunk", "M_meanmax", "M_v1", "L", "L_trunc128", "L_v1",
            "E", "E_unshrunk", "E_v1", "bm25", "tfidf", "dense_profile", "dense_trunc128",
            "F_rank", "F_v1", *[f"F_{n}" for n in k_variants], "F_zscore", "F_noL", "F_noE", "F_noM",
            "M_Score_Raw", "Context_Score", "R_Score_Raw", "Fair_Score"]
    df.assign(n_skills=df["skills"].str.len())[keep + ["n_skills"]].to_csv(out / "scores.csv", index=False)
    log.info(f"Scores, rankings and diagnostics written to {out} ({elapsed:.0f} s)")
    return diag


# ----------------------------------------------------------------------------- pool
def stage_pool(cfg, test=False):
    out = _out(cfg)
    R = json.loads((out / "rankings.json").read_text())["rankings"]
    df, _ = core.load_listings(cfg)
    cand = core.load_candidate(cfg)
    depth, n_rand = cfg["pool"]["depth"], cfg["pool"]["n_random"]
    sources = {}
    for name, r in R.items():
        for i in r[:depth]:
            sources.setdefault(i, []).append(name)
    rng = np.random.default_rng(cfg["seed"])
    rest = sorted(set(df["listing_id"]) - set(sources))
    for i in rng.choice(rest, size=min(n_rand, len(rest)), replace=False):
        sources.setdefault(str(i), []).append("random")
    pool = list(sources)
    rng.shuffle(pool)

    pdir = out / "pool"
    pdir.mkdir(exist_ok=True)
    by_id = df.set_index("listing_id")
    sheet = pd.DataFrame({"item": range(1, len(pool) + 1), "listing_id": pool,
                          "link": [by_id.loc[i, "link"] for i in pool],
                          "grade": "", "note": ""})
    for who in ("A", "B"):
        sheet.to_csv(pdir / f"labels_{who}.csv", index=False)
        _write_labeling_page(pdir / f"labeling_{who}.html", who, sheet, by_id, cand)
    pd.DataFrame({"listing_id": pool, "sources": [";".join(sources[i]) for i in pool]}).to_csv(
        pdir / "pool_sources_DO_NOT_OPEN_BEFORE_LABELING.csv", index=False)
    _write_guide(pdir / "LABELING_GUIDE.md", cand, len(pool))
    log.info(f"Judging pool: {len(pool)} listings ({len(pool) - min(n_rand, len(rest))} from system top-{depth}, "
             f"{min(n_rand, len(rest))} random) -> {pdir}")
    return len(pool)


GUIDE = """# Relevance labeling guide

You judge whether each volunteering listing is a suitable recommendation for ONE candidate,
whose profile is given below. Judge every listing on its own; the order is random.

Grades
- **2 - suitable**: the candidate meets the stated requirements and the activity matches his skills
  or the interests stated in the letter; you would recommend that he applies.
- **1 - partly suitable**: plausible but weak; either the activity fits his interests but needs skills
  he lacks, or it uses his skills but is far from his stated interests.
- **0 - not suitable**: unrelated to his skills and interests, or a stated eligibility requirement
  excludes him (country of residence, age range, required language, gender, profession).

Rules
- Ignore application deadlines and dates; judge as if the listing were open.
- Read the full description; listings in other languages are judged the same way.
- Do not look at any system output, score, or the pool_sources file until both annotators finish.
- Label independently; do not discuss listings with the other annotator before both are done.
- Use the note field for uncertain cases.

Pool size: {n} listings.

## Candidate profile

### Motivation letter
{letter}

### CV
{cv}
"""


def _write_guide(path, cand, n):
    path.write_text(GUIDE.format(n=n, letter=cand["letter"], cv=cand["cv_text"]), encoding="utf-8")


def _write_labeling_page(path, who, sheet, by_id, cand):
    items = [{"item": int(r.item), "id": r.listing_id, "link": r.link,
              "text": str(by_id.loc[r.listing_id, "description"])} for r in sheet.itertuples()]
    html = (Path(__file__).with_name("labeling_template.html").read_text(encoding="utf-8")
            .replace("__WHO__", who)
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__PROFILE__", json.dumps({"letter": cand["letter"], "cv": cand["cv_text"]},
                                               ensure_ascii=False).replace("</", "<\\/")))
    path.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------------- evaluate
def _read_labels(path):
    d = pd.read_csv(path, dtype={"listing_id": str})
    d = d[pd.to_numeric(d["grade"], errors="coerce").notna()].copy()
    d["grade"] = d["grade"].astype(float)
    bad = d[~d["grade"].isin([0, 1, 2])]
    if len(bad):
        raise ValueError(f"{path}: grades must be 0, 1 or 2 (rows {bad['item'].tolist()})")
    return d.set_index("listing_id")["grade"]


ANNOTATOR_NOTE = ("Annotator A saw the top-10 of the main system before labeling; "
                  "annotator B is the primary reference.")
GOLD_NAMES = {"B": "annotator B", "A": "annotator A", "mean": "mean of A and B"}
VALIDITY_SCORES = ["M", "M_unshrunk", "M_v1", "M_meanmax", "L", "L_trunc128", "E", "E_unshrunk", "E_v1",
                   "bm25", "tfidf", "dense_profile", "dense_trunc128", "F_rank", "F_v1", "F_zscore",
                   "M_Score_Raw", "Context_Score", "R_Score_Raw", "Fair_Score"]


def _evaluate_gold(gold, rankings, main, k, nb, seed, scores, random_ids):
    judged = gold.to_dict()
    point = metrics.evaluate_rankings(rankings, judged, k)
    ci = metrics.bootstrap(rankings, judged, k, nb, seed, reference=main)
    unjudged = {n: sum(1 for i in r[:k] if i not in judged) for n, r in rankings.items()}
    # Validity of each score: does it rank the judged listings like the labels do?
    S, g = scores.loc[list(judged)], pd.Series(judged)
    rnd = [i for i in random_ids if i in judged]
    return {
        "n_judged": len(judged),
        "grade_distribution": {str(x): int(c) for x, c in gold.value_counts().sort_index().items()},
        "systems": {n: {**point[n], **ci[n], f"unjudged_in_top{k}": unjudged[n]} for n in rankings},
        "score_validity": {c: {"spearman_with_grade": float(spearmanr(S[c], g).correlation),
                               "auc_relevant(grade>=1)": metrics.roc_auc(S[c], (g >= 1).astype(int))}
                           for c in VALIDITY_SCORES},
        "random_sample_relevant_share": {
            "n": len(rnd), "share_grade>=1": float(np.mean([judged[i] >= 1 for i in rnd])) if rnd else None}}


def stage_evaluate(cfg, test=False):
    from sklearn.metrics import cohen_kappa_score
    out = _out(cfg)
    pdir = out / "pool"
    primary = cfg.get("_gold", "B")
    R = json.loads((out / "rankings.json").read_text())
    rankings, main = R["rankings"], R["main"]
    pool_ids = pd.read_csv(pdir / "labels_A.csv", dtype={"listing_id": str})["listing_id"].tolist()
    A = _read_labels(pdir / "labels_A.csv")
    B = _read_labels(pdir / "labels_B.csv") if (pdir / "labels_B.csv").exists() else pd.Series(dtype=float)
    for who, lab in (("A", A), ("B", B)):
        if len(lab) < len(pool_ids):
            log.warning(f"Annotator {who} labeled {len(lab)}/{len(pool_ids)} listings")
    both = A.index.intersection(B.index)
    golds = {"B": B, "A": A, "mean": (A[both] + B[both]) / 2}
    golds = {n: g for n, g in golds.items() if len(g)}
    if primary not in golds:
        raise ValueError(f"--gold {primary}: the labels it needs are missing (labeled A {len(A)}, B {len(B)})")

    res = {"note": ANNOTATOR_NOTE, "primary_gold": primary, "n_pool": len(pool_ids),
           "n_labeled_A": int(len(A)), "n_labeled_B": int(len(B)),
           "agreement": {"n": int(len(both)),
                         "quadratic_weighted_kappa": float(cohen_kappa_score(A[both], B[both], weights="quadratic")),
                         "unweighted_kappa": float(cohen_kappa_score(A[both], B[both])),
                         "exact_agreement": float((A[both] == B[both]).mean())} if len(both) else
                        {"n": 0, "reason": "kappa needs listings labeled by both annotators"}}
    k, nb, seed = cfg["evaluation"]["k"], cfg["evaluation"]["n_bootstrap"], cfg["seed"]
    scores = pd.read_csv(out / "scores.csv", dtype={"listing_id": str}).set_index("listing_id")
    src = pd.read_csv(pdir / "pool_sources_DO_NOT_OPEN_BEFORE_LABELING.csv", dtype={"listing_id": str})
    random_ids = src[src["sources"].str.contains("random")]["listing_id"].tolist()
    order_ = [primary] + [n for n in GOLD_NAMES if n in golds and n != primary]
    res["by_gold"] = {n: _evaluate_gold(golds[n], rankings, main, k, nb, seed, scores, random_ids)
                      for n in order_}
    res["ILD@k"] = json.loads((out / "diagnostics.json").read_text())[f"ILD@{k}"]
    _dump(res, out / "results.json")
    (out / "results.md").write_text(_results_md(res, main, k), encoding="utf-8")
    log.info(f"Results written to {out / 'results.md'} (primary gold: {GOLD_NAMES[primary]})")
    return res


def _results_md(res, main, k):
    f = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
    lines = [res["note"], "",
             f"# Results (primary gold: {GOLD_NAMES[res['primary_gold']]}; pool {res['n_pool']}; "
             f"labeled A {res['n_labeled_A']}, B {res['n_labeled_B']})", ""]
    a = res["agreement"]
    lines += [f"Agreement on {a['n']} listings: quadratic-weighted kappa {a['quadratic_weighted_kappa']:.3f}, "
              f"unweighted kappa {a['unweighted_kappa']:.3f}, exact {a['exact_agreement']:.3f}"
              if a["n"] else f"Agreement: not computed ({a['reason']}).", ""]
    for i, (gname, blk) in enumerate(res["by_gold"].items()):
        title = "Main table" if i == 0 else "Additional table"
        lines += [f"## {title} (gold: {GOLD_NAMES[gname]}, {blk['n_judged']} judged listings)", "",
                  f"| System | NDCG@{k} | 95% CI | P@{k} | 95% CI | ΔNDCG vs main [95% CI] | ILD@{k} | unjudged |",
                  "|---|---|---|---|---|---|---|---|"]
        for n, s in sorted(blk["systems"].items(), key=lambda x: -x[1]["ndcg"]):
            d = s.get(f"ndcg_diff_vs_{main}")
            dpt = s["ndcg"] - blk["systems"][main]["ndcg"]
            dtxt = "reference" if n == main else (f"{dpt:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "")
            lines.append(f"| {n} | {s['ndcg']:.3f} | {f(s['ndcg_ci'])} | {s['p']:.2f} | {f(s['p_ci'])} | {dtxt} | "
                         f"{res['ILD@k'][n]:.3f} | {s[f'unjudged_in_top{k}']} |")
        lines += ["", "| Score | Spearman with grade | ROC-AUC (grade ≥ 1) |", "|---|---|---|"]
        for c, v in blk["score_validity"].items():
            lines.append(f"| {c} | {v['spearman_with_grade']:.3f} | {v['auc_relevant(grade>=1)']:.3f} |")
        r = blk["random_sample_relevant_share"]
        if r["n"]:
            lines += ["", f"Random sample: {r['share_grade>=1']:.2f} of {r['n']} listings have grade ≥ 1."]
        lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sanity", "score", "pool", "evaluate", "all"])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--test", action="store_true", help="hashing encoder + lexical NLI (no downloads)")
    ap.add_argument("--gold", choices=list(GOLD_NAMES), default="B",
                    help="evaluate: labels of the main table (A and mean are reported as additional tables)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg["_gold"] = a.gold
    stages = ["sanity", "score", "pool"] if a.stage == "all" else [a.stage]
    for s in stages:
        globals()[f"stage_{s}"](cfg, a.test)


if __name__ == "__main__":
    main()
