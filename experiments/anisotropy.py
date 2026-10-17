"""Outlier dimension in the reported run's skill vectors (xlm-roberta-base, mean pooling without padding mask).

  python -m experiments.anisotropy          # writes experiments/out/anisotropy.json

Vectors come from xlm-roberta-base with mean pooling over every position including padding
(backend hf_meanpool_nomask), before L2 normalization, from three sources:
  stored     - vectors that the reported run saved in data/raw/youth_portal_projeler_detayli.csv
               (column beceri_vektorleri). The column holds numpy's printed form, which numpy
               abbreviates with "..." once an array has more than 1,000 values, so complete
               768-value vectors exist only for listings with exactly one extracted skill.
               All-zero placeholders (listings without skills) are skipped.
  reencoded  - listings of the same file drawn at random (fixed seed) until at least 2,000 skills are
               collected; each listing's skill list is encoded as one batch, as the reported run did
               (src/skill_extraction/vectorizer.py), so padding enters the mean in the same way.
               The values that numpy did print for abbreviated rows are used to check that the
               re-encoding reproduces the stored vectors.
  esco       - both sides of the ESCO pairs (experiments/esco_pairs.csv), encoded in the same order and
               batches as `run.py sanity_esco`, so that the full-vector AUC reproduces sanity_esco.json.
The outlier dimension is fixed on the stored vectors (most frequent index of the largest absolute
coordinate) and then applied unchanged to the other sources; nothing is chosen on ESCO.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import core, metrics
from .encoders import _device

log = logging.getLogger("experiments")
ROOT = core.ROOT
STORED = ROOT / "data" / "raw" / "youth_portal_projeler_detayli.csv"
ESCO_PAIRS = ROOT / "experiments" / "esco_pairs.csv"
DIM = 768
VECTORIZER_BATCH = 32   # batch size of src/skill_extraction/vectorizer.py in the reported run
ESCO_BATCH = 64         # batch size of experiments/config.yaml (encoders.batch_size), as in sanity_esco


def _parse_skills(raw: str) -> list:
    raw = raw.strip()
    return [str(s) for s in ast.literal_eval(raw)] if raw.startswith("[") else []


def _parse_printed(raw: str):
    """Rows of a printed numpy array. Abbreviated rows contain '...'; returns (rows, abbreviated)."""
    body = raw.strip()[1:-1]                      # outer brackets
    rows = re.findall(r"\[([^\[\]]*)\]", body) if body.lstrip().startswith("[") else [body]
    abbreviated = "..." in raw
    return [r.split() for r in rows], abbreviated


def load_stored(path: Path = STORED):
    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df["skill_list"] = df["cikarilan_beceriler"].map(_parse_skills)
    vecs, info = [], {"rows": len(df), "rows_abbreviated_by_numpy": 0, "rows_complete": 0,
                      "zero_vectors_skipped": 0, "rows_with_unexpected_shape": 0}
    for raw in df["beceri_vektorleri"]:
        rows, abbreviated = _parse_printed(raw)
        if abbreviated:
            info["rows_abbreviated_by_numpy"] += 1
            continue
        info["rows_complete"] += 1
        a = np.array([x for r in rows for x in r], dtype=np.float64)
        if a.size % DIM:
            info["rows_with_unexpected_shape"] += 1
            continue
        for v in a.reshape(-1, DIM):
            if np.any(v):
                vecs.append(v)
            else:
                info["zero_vectors_skipped"] += 1
    info["complete_nonzero_vectors"] = len(vecs)
    return df, np.vstack(vecs), info


class RawNoMaskXLMR:
    """xlm-roberta-base, mean over all positions including padding, not normalized (the reported run)."""

    def __init__(self, name="xlm-roberta-base", device="auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.dev = _device(device)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.dev).eval()

    def batch(self, texts) -> np.ndarray:
        import torch
        enc = self.tok([str(t) for t in texts], padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to(self.dev)
        with torch.no_grad():
            return self.model(**enc).last_hidden_state.mean(dim=1).float().cpu().numpy().astype(np.float64)

    def per_listing(self, skills) -> np.ndarray:
        """As vectorize_skills: consecutive batches of 32 skills of one listing."""
        return np.vstack([self.batch(skills[i:i + VECTORIZER_BATCH]) for i in range(0, len(skills), VECTORIZER_BATCH)])

    def cached(self, texts, cache: dict) -> np.ndarray:
        """As encoders.Encoder.encode: distinct uncached texts in first-seen order, batches of 64."""
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(missing), ESCO_BATCH):
            b = missing[i:i + ESCO_BATCH]
            cache.update(zip(b, self.batch(b)))
        return np.stack([cache[t] for t in texts])


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def q(x):
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}


def describe(X, d, rng, n_pairs=2000):
    """Largest-|coordinate| statistics, and cosines of random pairs with and without dimension d."""
    A = np.abs(X)
    arg = A.argmax(axis=1)
    idx, cnt = np.unique(arg, return_counts=True)
    top = np.argsort(-cnt)[:5]
    mean_abs = A.mean(axis=0)
    by_mean = np.argsort(-mean_abs)[:5]
    others = np.delete(A, d, axis=1)
    i, j = rng.integers(0, len(X), size=(2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    cos = lambda Y: (unit(Y)[i] * unit(Y)[j]).sum(1)
    return {
        "n_vectors": int(len(X)),
        "largest_abs_dimension_counts": {int(idx[k]): int(cnt[k]) for k in top},
        "share_with_largest_abs_at_d": float(np.mean(arg == d)),
        "dimensions_with_largest_mean_abs": {int(k): float(mean_abs[k]) for k in by_mean},
        "value_at_d": q(X[:, d]),
        "median_abs_other_dimensions": q(np.median(others, axis=1)),
        "max_abs_other_dimensions": q(others.max(axis=1)),
        "share_of_squared_norm_at_d": q(X[:, d] ** 2 / (X ** 2).sum(1)),
        "norm": q(np.linalg.norm(X, axis=1)),
        "random_pairs": {"n": int(len(i)), "cosine_all_dimensions": q(cos(X)),
                         "cosine_without_d": q(cos(np.delete(X, d, axis=1)))},
    }


def reencode(df, enc, n_target, seed):
    """Random listings with skills until n_target skills; returns vectors, skills and a reproduction check
    against the values numpy printed for the same listings."""
    rows = df.index[df["skill_list"].str.len() > 0].to_numpy()
    order_ = np.random.default_rng(seed).permutation(rows)
    vecs, skills, diffs, n_rows, per_listing_max = [], [], [], 0, []
    for r in order_:
        sk = df.at[r, "skill_list"]
        V = enc.per_listing(sk)
        n_rows += 1
        printed, abbreviated = _parse_printed(df.at[r, "beceri_vektorleri"])
        d = _printed_diffs(V, printed, abbreviated)
        diffs.extend(d)
        if d:
            per_listing_max.append(max(d))
        vecs.append(V)
        skills.extend(sk)
        if len(skills) >= n_target:
            break
    check = {"listings_compared": n_rows, "printed_values_compared": len(diffs),
             "max_abs_difference": float(np.max(diffs)) if diffs else None,
             "median_abs_difference": float(np.median(diffs)) if diffs else None,
             "listings_with_all_printed_values_within_1e-4": int(sum(m <= 1e-4 for m in per_listing_max)),
             "printed_values_differing_by_more_than_1e-4": int(sum(x > 1e-4 for x in diffs))}
    return np.vstack(vecs), skills, n_rows, check


def _printed_diffs(V, printed, abbreviated):
    """|re-encoded - printed| for every value numpy printed (first/last 3 rows and columns when abbreviated)."""
    out = []
    if not abbreviated:
        a = np.array([x for r in printed for x in r], dtype=np.float64)
        if a.size == V.size:
            out.extend(np.abs(a - V.ravel()).tolist())
        return out
    n = len(V)
    row_idx = list(range(n)) if len(printed) == n else list(range(3)) + list(range(n - 3, n))
    for ri, toks in zip(row_idx, printed):
        if "..." in toks:
            k = toks.index("...")
            left, right = toks[:k], toks[k + 1:]
            cols = list(range(len(left))) + list(range(DIM - len(right), DIM))
            vals = left + right
        else:
            cols, vals = list(range(len(toks))), toks
        if len(vals) == len(cols) and all(re.fullmatch(r"-?[\d.]+(e[-+]\d+)?", v) for v in vals):
            out.extend(np.abs(np.array(vals, float) - V[ri, cols]).tolist())
    return out


def esco_auc(enc, d, n_boot, seed):
    pairs = pd.read_csv(ESCO_PAIRS)
    cache: dict = {}
    a = enc.cached(pairs.skill_a.astype(str).tolist(), cache)
    b = enc.cached(pairs.skill_b.astype(str).tolist(), cache)
    y = pairs.related.to_numpy()
    out = {"n_pairs": int(len(pairs))}
    for name, (A, B) in {"all_dimensions": (a, b),
                         "without_d": (np.delete(a, d, axis=1), np.delete(b, d, axis=1))}.items():
        cos = (unit(A) * unit(B)).sum(1)
        pos, neg = cos[y == 1], cos[y == 0]
        rng = np.random.default_rng(seed)
        boots = [metrics.roc_auc(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                                 np.r_[np.ones(len(pos)), np.zeros(len(neg))]) for _ in range(n_boot)]
        out[name] = {"auc_related_vs_unrelated": float(metrics.roc_auc(cos, y)),
                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                     "median_cosine_related": float(np.median(pos)),
                     "median_cosine_unrelated": float(np.median(neg))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--n-reencoded", type=int, default=2000)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    seed = cfg["seed"]

    df, X, info = load_stored()
    d = int(np.bincount(np.abs(X).argmax(axis=1), minlength=DIM).argmax())
    log.info(f"{len(X)} complete stored vectors; outlier dimension {d}")
    res = {"description": __doc__.strip().splitlines()[0], "outlier_dimension": d,
           "outlier_rule": "most frequent index of the largest absolute coordinate over the complete stored vectors",
           "stored_source": str(STORED.relative_to(ROOT)), "stored_parse": info,
           "stored": describe(X, d, np.random.default_rng(seed))}

    enc = RawNoMaskXLMR(device=cfg["encoders"]["device"])
    R, skills, n_rows, check = reencode(df, enc, a.n_reencoded, seed)
    res["reencoded"] = {"listings": n_rows, "distinct_skills": len(set(skills)),
                        "reproduction_of_printed_stored_values": check,
                        **describe(R, d, np.random.default_rng(seed))}
    log.info(f"re-encoded {len(R)} skills from {n_rows} listings; reproduction {check}")
    res["esco"] = esco_auc(enc, d, a.n_bootstrap, seed)
    res["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "encoder": "xlm-roberta-base [hf_meanpool_nomask, unnormalized]", "seed": seed}
    out = ROOT / cfg["paths"]["out_dir"] / "anisotropy.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info(f"written {out}")


if __name__ == "__main__":
    main()
