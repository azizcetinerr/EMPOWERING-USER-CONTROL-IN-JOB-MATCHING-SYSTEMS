"""Pair AUCs of the reported run's encoding with and without the outlier dimension (Table 2, last row).

  python -m experiments.dim741_pair_auc     # writes experiments/out/dim741_pair_auc.json

Encoding: xlm-roberta-base, mean over all positions including padding, unnormalized
(backend hf_meanpool_nomask), as in experiments/anisotropy.py. The outlier dimension is read from
experiments/out/anisotropy.json, where it is fixed on the stored vectors; nothing is chosen here.
For each pair set, the dimension is deleted, the vectors are L2-normalized, and the cosine is computed.
  esco - experiments/esco_pairs.csv, encoded as `run.py sanity_esco` (skill_a then skill_b, one cache,
         batches of 64), so the full-vector AUC reproduces sanity_esco.json.
  hand - experiments/skill_pairs.csv, encoded as `run.py sanity` (skill_a then skill_b, one cache,
         batches of 64), so the full-vector AUC reproduces sanity.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import core, metrics
from .anisotropy import ESCO_PAIRS, RawNoMaskXLMR, unit
from .run import ESCO_COMPARISONS

log = logging.getLogger("experiments")
ROOT = core.ROOT


def cosines(enc, pairs: pd.DataFrame, d: int | None) -> np.ndarray:
    cache: dict = {}
    a = enc.cached(pairs.skill_a.astype(str).tolist(), cache)
    b = enc.cached(pairs.skill_b.astype(str).tolist(), cache)
    if d is not None:
        a, b = np.delete(a, d, axis=1), np.delete(b, d, axis=1)
    return (unit(a) * unit(b)).sum(1)


def esco_aucs(cos: np.ndarray, pairs: pd.DataFrame) -> dict:
    t = pairs.pair_type.to_numpy()
    out = {}
    for name, (pos, neg) in ESCO_COMPARISONS.items():
        m = np.isin(t, pos + neg)
        out[name] = float(metrics.roc_auc(cos[m], np.isin(t[m], pos).astype(int)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["paths"]["out_dir"]
    d = int(json.loads((out_dir / "anisotropy.json").read_text())["outlier_dimension"])
    log.info(f"outlier dimension {d} (from anisotropy.json)")

    enc = RawNoMaskXLMR(device=cfg["encoders"]["device"])
    esco = pd.read_csv(ESCO_PAIRS)
    hand = pd.read_csv(ROOT / cfg["paths"]["skill_pairs"])

    res = {"encoder": "xlm-roberta-base", "backend": "hf_meanpool_nomask", "removed_dimension": d,
           "procedure": "mean over all positions including padding, unnormalized; delete dimension; "
                        "then L2-normalize and cosine",
           "esco_full": esco_aucs(cosines(enc, esco, None), esco),
           "esco_without_d": esco_aucs(cosines(enc, esco, d), esco),
           "hand_full_auc": float(metrics.roc_auc(cosines(enc, hand, None), hand.related)),
           "hand_without_d_auc": float(metrics.roc_auc(cosines(enc, hand, d), hand.related))}

    # reproduction checks against the stored encoder-check outputs (full vectors)
    checks = {}
    se = out_dir / "sanity_esco.json"
    if se.exists():
        rows = json.loads(se.read_text())["encoders"]
        ref = next((r for r in rows if r.get("backend") == "hf_meanpool_nomask"), None)
        if ref:
            checks["esco_full_vs_sanity_esco"] = {k: [res["esco_full"][k], ref["comparisons"][k]["auc"]]
                                                  for k in res["esco_full"]}
    sh = out_dir / "sanity.json"
    if sh.exists():
        rows = json.loads(sh.read_text())["encoders"]
        ref = next((r for r in rows if r.get("backend") == "hf_meanpool_nomask"), None)
        if ref:
            checks["hand_full_vs_sanity"] = [res["hand_full_auc"], ref["auc"]]
    res["checks"] = checks

    out = out_dir / "dim741_pair_auc.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info(f"written {out}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
