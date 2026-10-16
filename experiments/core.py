"""Data loading, the corrected scores, baselines, fusion and MMR."""
from __future__ import annotations

import ast
import logging
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("experiments")
ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- data
def clean_skills(skills, stopwords, min_length):
    out = []
    for s in skills:
        s = re.sub(r"\s+", " ", str(s).lower()).strip()
        if len(s) >= min_length and s not in stopwords and s not in out:
            out.append(s)
    return out


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def stopword_lists(langs) -> dict:
    import importlib
    return {l: importlib.import_module(f"spacy.lang.{l}.stop_words").STOP_WORDS for l in langs}


def detect_language(text, stops) -> str:
    """Language whose stopword list covers most words of the text ("unknown" for empty text)."""
    words = _WORD.findall(str(text).lower())
    if not words:
        return "unknown"
    return max(stops, key=lambda l: sum(w in stops[l] for w in words))


def filter_listing_skills(skill_lists, langs, stops, drop_stopwords=True, require_english=True):
    """SkillNER is English-only; on other languages it returns function words ("los", "en", "ses").
    Rule 1 drops skills that are a stopword in any listed language. Rule 2 keeps, for non-English
    listings, only skills that were also extracted from at least one English listing."""
    allstop = set().union(*stops.values()) if drop_stopwords else set()
    lists = [[s for s in lst if s not in allstop] for lst in skill_lists]
    if require_english:
        english = {s for lst, lg in zip(lists, langs) if lg == "en" for s in lst}
        lists = [lst if lg == "en" else [s for s in lst if s in english] for lst, lg in zip(lists, langs)]
    return lists


def load_listings(cfg) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(ROOT / cfg["paths"]["listings"], encoding="utf-8-sig")
    n_records = len(df)
    df = df.drop_duplicates("link").reset_index(drop=True)
    df["listing_id"] = df["link"].str.extract(r"/(\d+)_")[0]
    sk = cfg["skills"]
    df["skills"] = df["extracted_skills"].apply(
        lambda x: clean_skills(ast.literal_eval(x) if isinstance(x, str) else [],
                               set(sk["stopwords"]), sk["min_length"]))
    df["description"] = df["full_description"].fillna("").astype(str)
    n_before = int(df["skills"].str.len().sum())
    empty_before = int((df["skills"].str.len() == 0).sum())
    stops = stopword_lists(sk.get("stopword_languages", ["en"]))
    df["lang"] = df["description"].map(lambda t: detect_language(t, stops))
    df["skills"] = filter_listing_skills(df["skills"], df["lang"], stops,
                                         sk.get("drop_multilingual_stopwords", False),
                                         sk.get("non_english_requires_english_attestation", False))
    info = {"records": n_records, "distinct_listings": len(df),
            "duplicated_records_removed": n_records - len(df),
            "non_english_listings": int((df["lang"] != "en").sum()),
            "listing_skills_removed_by_language_filter": n_before - int(df["skills"].str.len().sum()),
            "listings_without_skills_before_language_filter": empty_before,
            "listings_without_skills": int((df["skills"].str.len() == 0).sum()),
            "listings_sharing_a_description": int(df["description"].duplicated(keep=False).sum())}
    return df, info


def load_candidate(cfg) -> dict:
    path = ROOT / cfg["paths"]["cv_text"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Save the full Europass CV as plain text there "
            "(one paragraph per line) before running this stage.")
    text = path.read_text(encoding="utf-8")
    min_chars = cfg["candidate"]["min_paragraph_chars"]
    paragraphs = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > min_chars]
    return {"cv_text": text, "paragraphs": paragraphs, "letter": cfg["candidate"]["letter"]}


def extract_cv_skills(cfg, cv_text: str) -> list[str]:
    """CV skills with the same SkillNER extractor that produced the listing skills.
    data/candidate/skills.txt, if present, overrides the extraction (one skill per line)."""
    override = ROOT / "data" / "candidate" / "skills.txt"
    sk = cfg["skills"]
    if override.exists():
        log.info(f"Using CV skills from {override}")
        raw = [ln for ln in override.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.skill_extraction.skill_extractor import SkillExtractorModule
        from src.utils.config_loader import load_config
        base = load_config(str(ROOT / "config" / "config.yaml"))
        base.setdefault("models", {}).setdefault("skill_extraction", {})["max_text_length"] = 0
        raw = SkillExtractorModule(base, log).extract_skills_from_text(cv_text)
    return clean_skills(raw, set(sk["stopwords"]), sk["min_length"])


# ----------------------------------------------------------------------------- scores
def idf_weights(skill_lists) -> dict:
    n = len(skill_lists)
    df = Counter(s for lst in skill_lists for s in set(lst))
    return {s: math.log((n + 1) / (c + 1)) + 1.0 for s, c in df.items()}


def best_skill_similarity(enc, listing_skill_lists, cv_skills) -> dict:
    """Highest cosine of each distinct listing skill to any CV skill."""
    uniq = sorted({s for lst in listing_skill_lists for s in lst})
    if not uniq or not cv_skills:
        return {s: 0.0 for s in uniq}
    P = enc.encode(uniq, "query")
    U = enc.encode(cv_skills, "query")
    best = (P @ U.T).max(axis=1)
    return dict(zip(uniq, best.astype(float)))


def matching_scores(skill_lists, best_sim, idf, tau_low, tau_high, use_idf=True):
    """M (soft, IDF-weighted coverage) and M_meanmax (original formula, same encoder)."""
    span = max(tau_high - tau_low, 1e-6)
    m_cov, m_mm = [], []
    for lst in skill_lists:
        if not lst:
            m_cov.append(0.0); m_mm.append(0.0); continue
        s = np.array([best_sim[x] for x in lst])
        w = np.array([idf[x] if use_idf else 1.0 for x in lst])
        c = np.clip((s - tau_low) / span, 0.0, 1.0)
        m_cov.append(100 * float((w * c).sum() / w.sum()))
        m_mm.append(100 * float(s.mean()))
    return np.array(m_cov), np.array(m_mm)


def shrink_to_median(scores, skill_lists, idf, k, use_idf=True) -> np.ndarray:
    """(n * score + k * median) / (n + k), n = IDF weight of the listing's skills in units of the mean
    skill weight. Few-skill listings move toward the median of listings with skills; empty ones get it."""
    scores = np.asarray(scores, dtype=float)
    has = np.array([len(l) > 0 for l in skill_lists])
    weight = lambda s: idf[s] if use_idf else 1.0
    occ = [weight(s) for l in skill_lists for s in l]
    w_mean = float(np.mean(occ)) if occ else 1.0
    n = np.array([sum(weight(s) for s in l) / w_mean for l in skill_lists])
    med = float(np.median(scores[has])) if has.any() else 0.0
    denom = n + k
    return np.where(denom > 0, (n * scores + k * med) / np.where(denom > 0, denom, 1.0), med)


def max_chunk_cosine(q, chunk_vecs) -> np.ndarray:
    """Highest cosine of the (1, dim) query to any chunk of each document."""
    return np.array([float((c @ q.ravel()).max()) for c in chunk_vecs])


def mean_chunk_embedding(chunk_vecs) -> np.ndarray:
    m = np.stack([c.mean(axis=0) for c in chunk_vecs])
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def skill_evidence(nli, skills, paragraphs, hypothesis_tpl) -> dict:
    """Highest entailment probability of "I have the skill of s." over all CV paragraphs."""
    skills = sorted(set(skills))
    if not paragraphs:
        return {s: 0.0 for s in skills}
    prem, hyp = [], []
    for s in skills:
        h = hypothesis_tpl.format(skill=s)
        prem += paragraphs
        hyp += [h] * len(paragraphs)
    p = nli.entail(prem, hyp).reshape(len(skills), len(paragraphs))
    return dict(zip(skills, p.max(axis=1).astype(float)))


def evidence_scores(skill_lists, evidence, idf, use_idf=True) -> np.ndarray:
    out = []
    for lst in skill_lists:
        if not lst:
            out.append(0.0); continue
        e = np.array([evidence[x] for x in lst])
        w = np.array([idf[x] if use_idf else 1.0 for x in lst])
        out.append(100 * float((w * e).sum() / w.sum()))
    return np.array(out)


# ----------------------------------------------------------------------------- baselines
_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _tokens(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [t for t in _TOKEN.findall(str(text).lower()) if t not in ENGLISH_STOP_WORDS]


def bm25_scores(query, docs) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([_tokens(d) for d in docs])
    return np.asarray(bm.get_scores(_tokens(query)), dtype=float)


def tfidf_scores(query, docs) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(tokenizer=_tokens, lowercase=False, token_pattern=None, sublinear_tf=True)
    X = vec.fit_transform(list(docs))
    q = vec.transform([query])
    return (X @ q.T).toarray().ravel()


# ----------------------------------------------------------------------------- fusion, MMR
def normalize(x, method):
    x = pd.Series(np.asarray(x, dtype=float))
    if method == "rank":
        return (100 * (x.rank(method="average") - 1) / max(len(x) - 1, 1)).to_numpy()
    if method == "zscore":   # the reported run's mapping
        sd = x.std()
        if not sd or np.isnan(sd):
            return np.full(len(x), 50.0)
        return (50 + 20 * (x - x.mean()) / sd).clip(0, 100).to_numpy()
    raise ValueError(method)


def fuse(scores: dict, weights: dict, method: str) -> np.ndarray:
    total = sum(weights.values())
    return sum(w / total * normalize(scores[k], method) for k, w in weights.items())


def order(scores) -> np.ndarray:
    """Positions sorted by descending score; ties broken by position (stable)."""
    s = np.asarray(scores, dtype=float)
    return np.argsort(-s, kind="stable")


def mmr_order(scores, emb, k, lam) -> np.ndarray:
    """Re-select the first k positions with MMR; the rest follow the score order.
    rel is the score min-max scaled to [0, 1]; sim is the cosine of description embeddings."""
    base = order(scores)
    if lam >= 1.0:
        return base
    s = np.asarray(scores, dtype=float)
    rel = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    selected = [int(base[0])]
    remaining = [int(i) for i in base[1:]]
    max_sim = emb[remaining] @ emb[selected[0]]
    while len(selected) < min(k, len(base)):
        val = lam * rel[remaining] - (1 - lam) * max_sim
        j = int(np.argmax(val))
        pick = remaining.pop(j)
        max_sim = np.delete(max_sim, j)
        selected.append(pick)
        if remaining:
            max_sim = np.maximum(max_sim, emb[remaining] @ emb[pick])
    return np.array(selected + remaining)
