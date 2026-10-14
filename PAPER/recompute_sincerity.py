"""Recomputes the sincerity (S) score with the documented formula for the 952 listings
and regenerates the z-scores, the weighted sum and the keyword bonus.

Why: the code that produced the S_Score_Raw column in IP5_final_fair_ranking.csv is not
in the repository. This script uses the sincerity_score.py and final_ranking.py classes of the
repository as they are; MMR and user feedback are not applied.

Usage (from the project root, in the environment where the pipeline runs; downloads models):
    python PAPER/recompute_sincerity.py
    python PAPER/recompute_sincerity.py --alpha 0.3 --beta 0.4   # config.yaml values
Then:
    python PAPER/verify_numbers.py outputs/results/IP5_S_recomputed.csv

Note: since S_human is the same for all listings, the z-scored S~ (and the ranking)
does not depend on the choice of alpha/beta; only the raw S values change.
"""
import argparse
import logging
import platform
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring.final_ranking import FinalRankingCalculator  # noqa: E402
from src.scoring.sincerity_score import SincerityScoreCalculator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4, help="S_human weight (code default 0.4)")
    ap.add_argument("--beta", type=float, default=None, help="S_context weight (default 1 - alpha)")
    ap.add_argument("--input", default="outputs/results/IP5_final_fair_ranking.csv")
    ap.add_argument("--output", default="outputs/results/IP5_S_recomputed.csv")
    args = ap.parse_args()
    beta = 1.0 - args.alpha if args.beta is None else args.beta

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("recompute_sincerity")

    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("sincerity_scoring", {})
    cfg["sincerity_scoring"]["human_score_weight"] = args.alpha
    cfg["sincerity_scoring"]["context_score_weight"] = beta

    df = pd.read_csv(ROOT / args.input)
    log.info(f"Input: {args.input} ({len(df)} listings); alpha={args.alpha}, beta={beta}")

    # --- IP3: S score (repository class, single letter from config.yaml)
    sc = SincerityScoreCalculator(cfg, log)
    log.info(f"P_AI (letter) = {sc.human_score:.6f}; device = {sc.device}")
    s_vals, h_vals, c_vals, times = [], [], [], []
    for i, text in enumerate(df["full_description"].astype(str)):
        t0 = time.perf_counter()
        s, h, c = sc.calculate_sincerity_score(text)
        times.append(time.perf_counter() - t0)
        s_vals.append(float(s)); h_vals.append(float(h)); c_vals.append(float(c))
        if (i + 1) % 100 == 0:
            log.info(f"  {i + 1}/{len(df)}")

    df["S_Score_Raw_old"] = df["S_Score_Raw"]
    df["S_Score_Raw"] = s_vals
    df["Human_Score"] = h_vals
    df["Context_Score"] = c_vals

    # --- IP5: z-score, weighted sum, keyword bonus (repository class)
    fr = FinalRankingCalculator(cfg, log)
    old_m, old_r = df["M_Score"].copy(), df["R_Score"].copy()
    for k in ["M", "S", "R"]:
        df[f"{k}_Score"] = fr.normalize_with_zscore(df[f"{k}_Score_Raw"], f"{k}_Score")
    df["Final_Score"] = [fr.calculate_final_score(m, s, r)
                        for m, s, r in zip(df.M_Score, df.S_Score, df.R_Score)]
    df["Fair_Score"] = df.apply(fr.apply_fairness_boost, axis=1) if fr.fairness_enabled else df["Final_Score"]
    dm, dr = (df["M_Score"] - old_m).abs().max(), (df["R_Score"] - old_r).abs().max()
    df = df.sort_values("Fair_Score", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / args.output, index=False)

    # --- Summary
    c = pd.Series(c_vals)
    print("\n=== SUMMARY ===")
    print(f"Environment: {platform.platform()}, Python {platform.python_version()}, device {sc.device}")
    print(f"P_AI = {sc.human_score:.6f}; S_human = {h_vals[0]:.4f}")
    print(f"S_context: mean {c.mean():.4f}, std {c.std():.4f}, min {c.min():.4f}, max {c.max():.4f}")
    print(f"S raw: mean {pd.Series(s_vals).mean():.4f}, std {pd.Series(s_vals).std():.4f}, "
          f"min {min(s_vals):.4f}, max {max(s_vals):.4f}")
    print(f"Max difference of M~ and R~ from the old CSV (should be 0): {dm:.2e}, {dr:.2e}")
    print(f"S computation time per listing: median {statistics.median(times) * 1000:.1f} ms, "
          f"mean {statistics.mean(times) * 1000:.1f} ms (n={len(times)})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
