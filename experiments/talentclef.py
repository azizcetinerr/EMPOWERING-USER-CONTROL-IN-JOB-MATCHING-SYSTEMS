"""External validation of the matching scores on TalentCLEF 2026 Task A (English, development set).

  python -m experiments.talentclef          # writes experiments/out/talentclef/
  python -m experiments.talentclef --test   # hashing encoders, lexical NLI, token skills (code path only)

Direction as in FAIR-PASS: query = job posting, document = CV. Every setting comes from
experiments/config.yaml and the encoder check (sanity_esco.json, else sanity.json); nothing is
tuned on TalentCLEF. Main result: every query ranks all CVs of the corpus. Additional table:
each query re-ranks its candidate set = labeled CVs + BM25 top 100.

Systems
  BM25        job text -> CV text
  Dense       selected encoder; job = normalized mean of its chunks, CV = best chunk
  M_old       reported run: XLM-R mean pooling without mask, mean over job skills of the best cosine
  M_meanmax   selected encoder, same mean-of-max formula
  M           selected encoder, IDF-weighted soft coverage with tau from the encoder check
  E           NLI evidence: per job skill, highest entailment probability over the CV paragraphs
  M+E         per-query rank normalization, weights M : E from the config (0.5 : 0.2)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import core, metrics
from .encoders import NLIScorer, make_encoder

log = logging.getLogger("experiments")
ROOT = core.ROOT
DATA = ROOT / "data" / "external" / "talentclef" / "TaskA" / "development" / "en"
OUT = ROOT / "experiments" / "out" / "talentclef"
OLD_ENCODER = {"name": "xlm-roberta-base", "backend": "hf_meanpool_nomask"}
SYSTEMS = ["BM25", "Dense", "M_old", "M_meanmax", "M", "E", "M+E"]
COMPARISONS = [("M", "BM25"), ("M", "Dense"), ("M", "M_old")]
METRICS = ["ndcg@10", "map", "p@10", "recall@100"]


# ----------------------------------------------------------------------------- data
def load_task_a(d: Path = DATA):
    read = lambda p: p.read_text(encoding="utf-8")
    queries = {p.name: read(p) for p in sorted((d / "queries").iterdir()) if p.is_file()}
    corpus = {p.name: read(p) for p in sorted((d / "corpus").iterdir()) if p.is_file()}
    q = pd.read_csv(d / "qrels.tsv", sep="\t", header=None, names=["qid", "it", "doc", "rel"], dtype=str)
    q = q[q["rel"].astype(float) > 0]
    qrels = {qid: set(g["doc"]) for qid, g in q.groupby("qid")}
    return queries, corpus, qrels


def paragraphs(text, min_chars):
    return [ln.strip() for ln in text.splitlines() if len(ln.strip()) > min_chars]


def _skill_extractor(cfg, test):
    sk = cfg["skills"]
    if test:
        return lambda t: core.clean_skills(sorted(set(re.findall(r"[a-z]{6,}", t.lower())))[:30],
                                           set(sk["stopwords"]), sk["min_length"])
    import sys
    sys.path.insert(0, str(ROOT))
    from src.skill_extraction.skill_extractor import SkillExtractorModule
    from src.utils.config_loader import load_config
    base = load_config(str(ROOT / "config" / "config.yaml"))
    base.setdefault("models", {}).setdefault("skill_extraction", {})["max_text_length"] = 0
    module = SkillExtractorModule(base, log)
    return lambda t: core.clean_skills(module.extract_skills_from_text(t), set(sk["stopwords"]), sk["min_length"])


def cached_skills(texts: dict, extract, path: Path) -> dict:
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [k for k in texts if k not in cache]
    for i, k in enumerate(todo, 1):
        cache[k] = extract(texts[k])
        if i % 25 == 0 or i == len(todo):
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            log.info(f"skills extracted for {i}/{len(todo)} texts ({path.name})")
    return {k: cache[k] for k in texts}


def cached_entailment(nli, skills, paras, hypothesis, path: Path) -> dict:
    """{skill: P(entailment) per paragraph, aligned with `paras`}; saved after each skill so an
    interrupted run resumes. Paragraphs are fed in length order (less padding) and mapped back."""
    cache = pickle.loads(path.read_bytes()) if path.exists() else {}
    if cache.get("paras") != paras or cache.get("hypothesis") != hypothesis:
        cache = {"paras": paras, "hypothesis": hypothesis, "p": {}}
    order = np.argsort([len(p) for p in paras], kind="stable")
    sorted_paras = [paras[i] for i in order]
    todo = [s for s in skills if s not in cache["p"]]
    t0 = time.perf_counter()
    for i, s in enumerate(todo, 1):
        p = np.empty(len(paras), np.float32)
        p[order] = nli.entail(sorted_paras, [hypothesis.format(skill=s)] * len(paras))
        cache["p"][s] = p
        path.write_bytes(pickle.dumps(cache))
        if i % 10 == 0 or i == len(todo):
            log.info(f"NLI {i}/{len(todo)} skills ({(time.perf_counter() - t0) / i:.1f} s per skill)")
    return cache["p"]


# ----------------------------------------------------------------------------- metrics
def average_precision(ranking, relevant) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def query_metrics(ranking, relevant) -> dict:
    g = [1.0 if d in relevant else 0.0 for d in ranking]
    return {"ndcg@10": metrics.ndcg_at_k(g, [1.0] * len(relevant), 10),
            "map": average_precision(ranking, relevant),
            "p@10": metrics.precision_at_k(g, 10),
            "recall@100": sum(g[:100]) / len(relevant) if relevant else 0.0}


def paired_randomization(a, b, n_perm=10_000, seed=0) -> float:
    """Two-sided sign-flip test of the mean per-query difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def bootstrap_ci(diff, n_boot=10_000, seed=0) -> list:
    d = np.asarray(diff, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * pvals[k]))
        adj[k] = running
    return adj


# ----------------------------------------------------------------------------- scores
def skill_scores(q_skills, cv_skills, sim, idf, tau_low, tau_high):
    """mean-of-max and IDF-weighted soft coverage of the job skills by one CV's skills."""
    if not q_skills or not cv_skills:
        return 0.0, 0.0
    best = np.array([max(sim[(s, c)] for c in cv_skills) for s in q_skills])
    w = np.array([idf[s] for s in q_skills])
    cov = np.clip((best - tau_low) / max(tau_high - tau_low, 1e-6), 0.0, 1.0)
    return 100 * float(best.mean()), 100 * float((w * cov).sum() / w.sum())


def pair_similarity(enc, q_skills: set, cv_skills: set) -> dict:
    qs, cs = sorted(q_skills), sorted(cv_skills)
    S = enc.encode(qs, "query") @ enc.encode(cs, "query").T
    return {(q, c): float(S[i, j]) for i, q in enumerate(qs) for j, c in enumerate(cs)}


def score_all(cfg, queries, corpus, test=False):
    out = OUT if not test else OUT.parent / "talentclef_test"
    (out / "cache").mkdir(parents=True, exist_ok=True)
    exp_out = ROOT / cfg["paths"]["out_dir"]
    sfile = exp_out / ("sanity_esco.json" if (exp_out / "sanity_esco.json").exists() else "sanity.json")
    sanity = json.loads(sfile.read_text())
    new_spec = {"name": "hashing-a", "backend": "hashing"} if test else sanity["selected"]
    old_spec = {"name": "hashing-b", "backend": "hashing"} if test else OLD_ENCODER
    bs, dev = cfg["encoders"]["batch_size"], cfg["encoders"]["device"]

    extract = _skill_extractor(cfg, test)
    sfx = "_test" if test else ""
    qsk = cached_skills(queries, extract, out / "cache" / f"skills_queries{sfx}.json")
    csk = cached_skills(corpus, extract, out / "cache" / f"skills_cvs{sfx}.json")
    idf = core.idf_weights(list(qsk.values()))
    all_q = {s for l in qsk.values() for s in l}
    all_c = {s for l in csk.values() for s in l}

    t0 = time.perf_counter()
    enc = make_encoder(new_spec, bs, dev)
    sim_new = pair_similarity(enc, all_q, all_c)
    sim_old = pair_similarity(make_encoder(old_spec, bs, dev), all_q, all_c)
    ch = cfg["encoders"]["passage_chunks"]
    doc_ids = list(corpus)
    doc_chunks = enc.encode_chunked([corpus[d] for d in doc_ids], "passage", ch["tokens"], ch["overlap"])

    min_chars = cfg["candidate"]["min_paragraph_chars"]
    paras = {d: paragraphs(t, min_chars) for d, t in corpus.items()}
    uniq_paras = sorted({p for ps in paras.values() for p in ps})
    nli = NLIScorer(cfg["nli"]["model"], cfg["nli"]["batch_size"], dev, backend="lexical" if test else "hf")
    ent = cached_entailment(nli, sorted(all_q), uniq_paras, cfg["nli"]["hypothesis"],
                            out / "cache" / f"nli{sfx}.pkl")

    para_idx = {p: i for i, p in enumerate(uniq_paras)}
    doc_para = {d: np.array([para_idx[p] for p in ps], int) for d, ps in paras.items()}
    w = cfg["fusion"]["weights"]
    rows = []
    for qid, qtext in queries.items():
        qs = qsk[qid]
        qv = core.mean_chunk_embedding(enc.encode_chunked([qtext], "query", ch["tokens"], ch["overlap"]))
        bm25 = core.bm25_scores(qtext, [corpus[d] for d in doc_ids])
        dense = core.max_chunk_cosine(qv, doc_chunks)
        for j, d in enumerate(doc_ids):
            mm_new, m_cov = skill_scores(qs, csk[d], sim_new, idf, sanity["tau_low"], sanity["tau_high"])
            mm_old, _ = skill_scores(qs, csk[d], sim_old, idf, sanity["tau_low"], sanity["tau_high"])
            if qs and paras[d]:
                e = np.array([ent[s][doc_para[d]].max() for s in qs])
                wq = np.array([idf[s] for s in qs])
                e_score = 100 * float((wq * e).sum() / wq.sum())
            else:
                e_score = 0.0
            rows.append({"qid": qid, "doc": d, "BM25": bm25[j], "Dense": dense[j], "M_old": mm_old,
                         "M_meanmax": mm_new, "M": m_cov, "E": e_score})
    S = pd.DataFrame(rows)
    S["M+E"] = S.groupby("qid", group_keys=False).apply(
        lambda g: pd.Series(core.fuse({"M": g["M"], "E": g["E"]}, {"M": w["M"], "E": w["E"]}, "rank"), index=g.index))
    info = {"encoder": new_spec, "old_encoder": old_spec, "tau_source": sfile.name,
            "tau_low": sanity["tau_low"], "tau_high": sanity["tau_high"],
            "fusion_weights_M_E": {"M": w["M"], "E": w["E"]}, "nli": cfg["nli"]["model"],
            "passage_chunks": ch, "n_query_skills": {q: len(v) for q, v in qsk.items()},
            "cv_skills_median": float(np.median([len(v) for v in csk.values()])),
            "cvs_without_skills": int(sum(not v for v in csk.values())),
            "distinct_query_skills": len(all_q), "distinct_cv_skills": len(all_c),
            "cv_paragraphs_unique": len(uniq_paras), "scoring_seconds": time.perf_counter() - t0}
    return S, info, out


# ----------------------------------------------------------------------------- evaluation
def rank_docs(g: pd.DataFrame, system: str) -> list:
    return g["doc"].to_numpy()[core.order(g[system].to_numpy())].tolist()


def evaluate(S, qrels, n_perm=10_000, n_boot=10_000, seed=2026):
    per_query, settings = [], {}
    for setting in ("full_corpus", "candidate_set"):
        for qid, g in S.groupby("qid", sort=True):
            rel = qrels.get(qid, set())
            if setting == "candidate_set":
                top = set(rank_docs(g, "BM25")[:100])
                g = g[g["doc"].isin(rel | top)]
            for sysname in SYSTEMS:
                per_query.append({"setting": setting, "qid": qid, "system": sysname, "n_candidates": len(g),
                                  **query_metrics(rank_docs(g, sysname), rel)})
    P = pd.DataFrame(per_query)
    for setting, ps in P.groupby("setting", sort=False):
        means = ps.groupby("system")[METRICS].mean().reindex(SYSTEMS)
        comps = {}
        for m in METRICS:
            piv = ps.pivot(index="qid", columns="system", values=m)
            raw = {f"{a} vs {b}": paired_randomization(piv[a], piv[b], n_perm, seed) for a, b in COMPARISONS}
            adj = holm(raw)
            comps[m] = {f"{a} vs {b}": {"mean_diff": float((piv[a] - piv[b]).mean()),
                                        "ci95": bootstrap_ci(piv[a] - piv[b], n_boot, seed),
                                        "p": raw[f"{a} vs {b}"], "p_holm": adj[f"{a} vs {b}"]}
                        for a, b in COMPARISONS}
        settings[setting] = {"means": means.to_dict(orient="index"), "comparisons": comps,
                             "n_queries": int(ps["qid"].nunique()),
                             "candidates_per_query": ps.groupby("qid")["n_candidates"].first().to_dict()}
    return P, settings


def results_md(res) -> str:
    lines = ["# TalentCLEF 2026 Task A (English, development set): external validation", "",
             f"Queries (job postings): {res['data']['queries']}; CVs: {res['data']['cvs']}; "
             f"relevant pairs: {res['data']['relevant_pairs']} (binary labels). "
             f"Encoder: {res['scoring']['encoder']['name']}; tau {res['scoring']['tau_low']:.3f} / "
             f"{res['scoring']['tau_high']:.3f} from {res['scoring']['tau_source']}. Nothing tuned on TalentCLEF.", ""]
    titles = {"full_corpus": "Main table: every query ranks all CVs",
              "candidate_set": "Additional table: candidate set = labeled CVs + BM25 top 100"}
    for setting, blk in res["settings"].items():
        lines += [f"## {titles[setting]}", "", "| System | NDCG@10 | MAP | P@10 | Recall@100 |", "|---|---|---|---|---|"]
        for s in SYSTEMS:
            v = blk["means"][s]
            lines.append(f"| {s} | {v['ndcg@10']:.3f} | {v['map']:.3f} | {v['p@10']:.3f} | {v['recall@100']:.3f} |")
        lines += ["", "Paired comparisons over queries (randomization test, 10,000 sign flips; bootstrap 95% CI; "
                  "Holm correction over the three comparisons per metric):", "",
                  "| Metric | Comparison | Mean difference | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
        for m, comps in blk["comparisons"].items():
            for name, c in comps.items():
                lines.append(f"| {m} | {name} | {c['mean_diff']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                             f"{c['p']:.4f} | {c['p_holm']:.4f} |")
        lines.append("")
    lines += [f"Runtime {res['runtime_seconds']:.0f} s; {res['environment']}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dataset-version", default="Task A development set, files dated 2026-02-01 (TaskA.zip md5 a966109f7e6c2a6c30111fe814fe0bae); Zenodo version not yet confirmed")
    ap.add_argument("--archive-md5", default="a966109f7e6c2a6c30111fe814fe0bae", help="md5 of the downloaded TaskA.zip")
    ap.add_argument("--files-dated", default="2026-02-01", help="date of the files inside TaskA.zip")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    queries, corpus, qrels = load_task_a()
    S, info, out = score_all(cfg, queries, corpus, a.test)
    P, settings = evaluate(S, qrels, seed=cfg["seed"])
    res = {"data": {"source": "TalentCLEF 2026 Task A, development set, English (Zenodo 10.5281/zenodo.17625261)",
                    "license": "CC BY 4.0", "version": a.dataset_version, "queries": len(queries),
                    "cvs": len(corpus), "relevant_pairs": int(sum(len(v) for v in qrels.values())),
                    "relevant_per_query": {q: len(v) for q, v in qrels.items()},
                    "archive_md5": {"TaskA.zip": a.archive_md5}, "files_dated": a.files_dated},
           "scoring": info, "settings": settings, "runtime_seconds": time.perf_counter() - t0,
           "environment": f"python {platform.python_version()}, {platform.platform()}"}
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "scores.csv", index=False)
    P.to_csv(out / "per_query.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "results.md").write_text(results_md(res), encoding="utf-8")
    log.info(f"Results written to {out}")


if __name__ == "__main__":
    main()
