"""Makaledeki turetilmis sayilari skor dosyasindan yeniden hesaplar.

Kullanim (proje kokunden):
    python PAPER/verify_numbers.py outputs/results/IP5_S_yeniden.csv
Ikinci arguman (istege bagli) onceki surumun dosyasidir; verilmezse
outputs/results/IP5_Nihai_Adil_Siralama.csv varsa kullanilir.
Gereksinim: pandas, scipy
"""
import ast
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

W_M, W_S, W_R = 0.5, 0.3, 0.2

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else
                 "outputs/results/IP5_S_yeniden.csv")
print(f"Listings: {len(df)}")

# Sinirliliklar: tekrarlanan kayitlar
print(f"Duplicated links: {df.link.duplicated().sum()}; distinct listings: {df.link.nunique()}")

# Tablo I: ham ve standardize skorlar (orneklem std, ddof=1)
for raw, std in [("M_Skoru_Ham", "M_Skoru"), ("S_Skoru_Ham", "S_Skoru"),
                 ("R_Skoru_Ham", "R_Skoru")]:
    r, s = df[raw], df[std]
    print(f"{raw[0]} raw {r.mean():.2f} ({r.std():.2f}) {r.min():.2f}-{r.max():.2f} | "
          f"std {s.mean():.2f} ({s.std():.2f}) {s.min():.2f}-{s.max():.2f}")
for c in ["Final_Skor", "Adil_Skor"]:
    print(f"{c} {df[c].mean():.4f} ({df[c].std():.2f}) {df[c].min():.2f}-{df[c].max():.2f}")

# Beceri sayilari
n_sk = df["cikarilan_beceriler"].apply(
    lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
print(f"Listings with >=1 skill: {(n_sk > 0).sum()}; mean skills all {n_sk.mean():.2f}; "
      f"mean among >0 {n_sk[n_sk > 0].mean():.2f}")

# Bolum IV-A: ham M sikismasi
m = df["M_Skoru_Ham"]
nz = df[m > 0]
print(f"M==0: {(m == 0).sum()}; M>99: {(m > 99).sum()}; "
      f"nonzero M {nz.M_Skoru_Ham.min():.2f}-{nz.M_Skoru_Ham.max():.2f} sd {nz.M_Skoru_Ham.std():.3f}")

# Bolum IV-A: samimiyet bilesenleri
ctx = df["Baglam_Puani"]
print(f"S_human (Insan_Puani) unique: {sorted(df.Insan_Puani.round(5).unique())}")
print(f"S_context {ctx.mean():.2f} ({ctx.std():.2f}) {ctx.min():.2f}-{ctx.max():.2f}")
print("Spearman S_context vs description length (chars):",
      round(spearmanr(ctx, df.tam_aciklama.str.len()).correlation, 2))
print(f"Std. S clipped at 0: {(df.S_Skoru <= 0).sum()}")

# Bolum IV-B: katkilar
print("Raw contributions:", round(W_M * m.mean(), 2), round(W_S * df.S_Skoru_Ham.mean(), 2),
      round(W_R * df.R_Skoru_Ham.mean(), 2))
print("Std. contributions:", round(W_M * df.M_Skoru.mean(), 2), round(W_S * df.S_Skoru.mean(), 2),
      round(W_R * df.R_Skoru.mean(), 2))
print("SD of weighted contributions (931):", round((W_M * nz.M_Skoru).std(), 2),
      round((W_S * nz.S_Skoru).std(), 2), round((W_R * nz.R_Skoru).std(), 2))
print("Spearman A vs raw M / std S / std R (931):",
      round(spearmanr(nz.Adil_Skor, nz.M_Skoru_Ham).correlation, 2),
      round(spearmanr(nz.Adil_Skor, nz.S_Skoru).correlation, 2),
      round(spearmanr(nz.Adil_Skor, nz.R_Skoru).correlation, 2))
top_m = set(df.sort_values("M_Skoru_Ham", ascending=False).head(10).link)
top_a = df.sort_values("Adil_Skor", ascending=False).head(10)
top_f = df.sort_values("Final_Skor", ascending=False).head(10)
print("Top-10 overlap raw M vs A:", len(top_m & set(top_a.link)))
print("Top-10 identical for F and A:", list(top_f.link) == list(top_a.link))

# Tablo II
t = top_a.assign(id=top_a.link.str.extract(r"/(\d+)_")[0])
print(t[["id", "M_Skoru", "S_Skoru", "R_Skoru", "Adil_Skor"]].round(2).to_string(index=False))

# Bolum IV-C: anahtar kelime bonusu
print("Keyword bonus applied:", int(((df.Adil_Skor - df.Final_Skor).round(6) == 5).sum()))

# Depolama tahmini
print(f"Storage 952 x 768 x 4 B = {952 * 768 * 4 / 1e6:.2f} MB")

# Onceki surumle karsilastirma (IP5_Nihai_Adil_Siralama.csv)
old_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/results/IP5_Nihai_Adil_Siralama.csv"
if os.path.exists(old_path):
    old = pd.read_csv(old_path, encoding="utf-8-sig")
    new_u = df.drop_duplicates("link").set_index("link")["Adil_Skor"]
    old_u = old.drop_duplicates("link").set_index("link")["Adil_Skor"]
    common = new_u.index.intersection(old_u.index)
    shared = (set(df.sort_values("Adil_Skor", ascending=False).head(10).link)
              & set(old.sort_values("Adil_Skor", ascending=False).head(10).link))
    ids = sorted(int(l.split("/")[-1].split("_")[0]) for l in shared)
    print(f"Previous version: shared top-10 = {len(shared)} {ids}; "
          f"Spearman A_old vs A_new over {len(common)} distinct listings = "
          f"{spearmanr(old_u[common], new_u[common]).correlation:.2f}")
