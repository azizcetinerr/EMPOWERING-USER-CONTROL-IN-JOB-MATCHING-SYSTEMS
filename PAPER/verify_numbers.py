"""Recomputes the derived numbers reported in the paper from the score file.

Usage (from the project root):
    python PAPER/verify_numbers.py outputs/results/IP5_S_recomputed.csv
The second argument (optional) is the file of the earlier version; if it is not given,
outputs/results/IP5_final_fair_ranking.csv is used when it exists.
Requirements: pandas, scipy
"""
import ast
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

W_M, W_S, W_R = 0.5, 0.3, 0.2

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else
                 "outputs/results/IP5_S_recomputed.csv")
print(f"Listings: {len(df)}")

# Limitations: duplicated records
print(f"Duplicated links: {df.link.duplicated().sum()}; distinct listings: {df.link.nunique()}")

# Table I: raw and standardized scores (sample std, ddof=1)
for raw, std in [("M_Score_Raw", "M_Score"), ("S_Score_Raw", "S_Score"),
                 ("R_Score_Raw", "R_Score")]:
    r, s = df[raw], df[std]
    print(f"{raw[0]} raw {r.mean():.2f} ({r.std():.2f}) {r.min():.2f}-{r.max():.2f} | "
          f"std {s.mean():.2f} ({s.std():.2f}) {s.min():.2f}-{s.max():.2f}")
for c in ["Final_Score", "Fair_Score"]:
    print(f"{c} {df[c].mean():.4f} ({df[c].std():.2f}) {df[c].min():.2f}-{df[c].max():.2f}")

# Skill counts
n_sk = df["extracted_skills"].apply(
    lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
print(f"Listings with >=1 skill: {(n_sk > 0).sum()}; mean skills all {n_sk.mean():.2f}; "
      f"mean among >0 {n_sk[n_sk > 0].mean():.2f}")

# Section IV-A: compression of the raw M score
m = df["M_Score_Raw"]
nz = df[m > 0]
print(f"M==0: {(m == 0).sum()}; M>99: {(m > 99).sum()}; "
      f"nonzero M {nz.M_Score_Raw.min():.2f}-{nz.M_Score_Raw.max():.2f} sd {nz.M_Score_Raw.std():.3f}")

# Section IV-A: sincerity components
ctx = df["Context_Score"]
print(f"S_human (Human_Score) unique: {sorted(df.Human_Score.round(5).unique())}")
print(f"S_context {ctx.mean():.2f} ({ctx.std():.2f}) {ctx.min():.2f}-{ctx.max():.2f}")
print("Spearman S_context vs description length (chars):",
      round(spearmanr(ctx, df.full_description.str.len()).correlation, 2))
print(f"Std. S clipped at 0: {(df.S_Score <= 0).sum()}")

# Section IV-B: contributions
print("Raw contributions:", round(W_M * m.mean(), 2), round(W_S * df.S_Score_Raw.mean(), 2),
      round(W_R * df.R_Score_Raw.mean(), 2))
print("Std. contributions:", round(W_M * df.M_Score.mean(), 2), round(W_S * df.S_Score.mean(), 2),
      round(W_R * df.R_Score.mean(), 2))
print("SD of weighted contributions (931):", round((W_M * nz.M_Score).std(), 2),
      round((W_S * nz.S_Score).std(), 2), round((W_R * nz.R_Score).std(), 2))
print("Spearman A vs raw M / std S / std R (931):",
      round(spearmanr(nz.Fair_Score, nz.M_Score_Raw).correlation, 2),
      round(spearmanr(nz.Fair_Score, nz.S_Score).correlation, 2),
      round(spearmanr(nz.Fair_Score, nz.R_Score).correlation, 2))
top_m = set(df.sort_values("M_Score_Raw", ascending=False).head(10).link)
top_a = df.sort_values("Fair_Score", ascending=False).head(10)
top_f = df.sort_values("Final_Score", ascending=False).head(10)
print("Top-10 overlap raw M vs A:", len(top_m & set(top_a.link)))
print("Top-10 identical for F and A:", list(top_f.link) == list(top_a.link))

# Table II
t = top_a.assign(id=top_a.link.str.extract(r"/(\d+)_")[0])
print(t[["id", "M_Score", "S_Score", "R_Score", "Fair_Score"]].round(2).to_string(index=False))

# Section IV-C: keyword bonus
print("Keyword bonus applied:", int(((df.Fair_Score - df.Final_Score).round(6) == 5).sum()))

# Storage estimate
print(f"Storage 952 x 768 x 4 B = {952 * 768 * 4 / 1e6:.2f} MB")

# Comparison with the earlier version (IP5_final_fair_ranking.csv)
old_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/results/IP5_final_fair_ranking.csv"
if os.path.exists(old_path):
    old = pd.read_csv(old_path, encoding="utf-8-sig")
    new_u = df.drop_duplicates("link").set_index("link")["Fair_Score"]
    old_u = old.drop_duplicates("link").set_index("link")["Fair_Score"]
    common = new_u.index.intersection(old_u.index)
    shared = (set(df.sort_values("Fair_Score", ascending=False).head(10).link)
              & set(old.sort_values("Fair_Score", ascending=False).head(10).link))
    ids = sorted(int(l.split("/")[-1].split("_")[0]) for l in shared)
    print(f"Previous version: shared top-10 = {len(shared)} {ids}; "
          f"Spearman A_old vs A_new over {len(common)} distinct listings = "
          f"{spearmanr(old_u[common], new_u[common]).correlation:.2f}")
