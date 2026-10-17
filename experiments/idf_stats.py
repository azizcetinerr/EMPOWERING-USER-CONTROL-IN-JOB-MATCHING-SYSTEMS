"""Distribution of the IDF skill weights (core.idf_weights) on the volunteering snapshot and on TalentCLEF.

  python -m experiments.idf_stats      # writes experiments/out/idf_stats.json

Volunteering: skill lists of the main system (config_v3.yaml: language filter applied), N = listings.
TalentCLEF: skill lists of the 10 job postings (out/talentclef/cache/skills_queries.json, written by
talentclef.py), N = postings. No CV data is read.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from . import core

ROOT = core.ROOT
EXP = ROOT / "experiments"
TC_QUERIES = EXP / "out" / "talentclef" / "cache" / "skills_queries.json"


def idf_summary(skill_lists) -> dict:
    """Number of lists and distinct skills, document-frequency counts, and the distinct weight values."""
    idf = core.idf_weights(skill_lists)
    df = Counter(s for lst in skill_lists for s in set(lst))
    w = np.array(list(idf.values()))
    w_max = float(w.max()) if len(w) else 0.0
    return {"n_lists": len(skill_lists), "distinct_skills": len(df),
            "skills_by_document_frequency": {int(k): int(v) for k, v in sorted(Counter(df.values()).items())},
            "share_skills_in_one_list": float(np.mean([c == 1 for c in df.values()])) if df else 0.0,
            "distinct_weight_values": int(len(np.unique(np.round(w, 6)))),
            "weight_min": float(w.min()) if len(w) else 0.0, "weight_max": w_max,
            "share_skills_with_max_weight": float(np.isclose(w, w_max).mean()) if len(w) else 0.0}


def main():
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    cfg3 = yaml.safe_load((EXP / "config_v3.yaml").read_text(encoding="utf-8"))
    listings, _ = core.load_listings({**cfg3, "paths": cfg["paths"]})
    res = {"volunteering_v3": idf_summary(listings["skills"].tolist())}
    if TC_QUERIES.exists():
        q = json.loads(TC_QUERIES.read_text(encoding="utf-8"))
        res["talentclef_postings"] = idf_summary(list(q.values()))
    else:
        res["talentclef_postings"] = None
    res["formula"] = "w(s) = ln((N + 1) / (n_s + 1)) + 1 (core.idf_weights)"
    out = ROOT / cfg["paths"]["out_dir"] / "idf_stats.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
