"""Sincerity (S) skorunu belgelenen formulle 952 ilan icin yeniden hesaplar
ve z-score, agirlikli toplam ile anahtar kelime bonusunu yeniden uretir.

Neden: IP5_Nihai_Adil_Siralama.csv icindeki S_Skoru_Ham sutununu ureten kod
repoda yok. Bu betik, repodaki sincerity_score.py ve final_ranking.py
siniflarini oldugu gibi kullanir; MMR ve kullanici geri bildirimi uygulanmaz.

Kullanim (proje kokunden, pipeline'in calistigi ortamda; model indirme gerekir):
    python PAPER/recompute_sincerity.py
    python PAPER/recompute_sincerity.py --alpha 0.3 --beta 0.4   # config.yaml degerleri
Ardindan:
    python PAPER/verify_numbers.py outputs/results/IP5_S_yeniden.csv

Not: S_human tum ilanlar icin ayni oldugundan, z-score sonrasi S~ (ve siralama)
alpha/beta secimine bagli degildir; yalnizca ham S degerleri degisir.
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
    ap.add_argument("--alpha", type=float, default=0.4, help="S_human agirligi (kod varsayilani 0.4)")
    ap.add_argument("--beta", type=float, default=None, help="S_context agirligi (varsayilan 1 - alpha)")
    ap.add_argument("--input", default="outputs/results/IP5_Nihai_Adil_Siralama.csv")
    ap.add_argument("--output", default="outputs/results/IP5_S_yeniden.csv")
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
    log.info(f"Girdi: {args.input} ({len(df)} ilan); alpha={args.alpha}, beta={beta}")

    # --- IP3: S skoru (repodaki sinif, tek mektup config.yaml'dan)
    sc = SincerityScoreCalculator(cfg, log)
    log.info(f"P_AI (mektup) = {sc.human_score:.6f}; cihaz = {sc.device}")
    s_vals, h_vals, c_vals, times = [], [], [], []
    for i, text in enumerate(df["tam_aciklama"].astype(str)):
        t0 = time.perf_counter()
        s, h, c = sc.calculate_sincerity_score(text)
        times.append(time.perf_counter() - t0)
        s_vals.append(float(s)); h_vals.append(float(h)); c_vals.append(float(c))
        if (i + 1) % 100 == 0:
            log.info(f"  {i + 1}/{len(df)}")

    df["S_Skoru_Ham_eski"] = df["S_Skoru_Ham"]
    df["S_Skoru_Ham"] = s_vals
    df["Insan_Puani"] = h_vals
    df["Baglam_Puani"] = c_vals

    # --- IP5: z-score, agirlikli toplam, anahtar kelime bonusu (repodaki sinif)
    fr = FinalRankingCalculator(cfg, log)
    old_m, old_r = df["M_Skoru"].copy(), df["R_Skoru"].copy()
    for k in ["M", "S", "R"]:
        df[f"{k}_Skoru"] = fr.normalize_with_zscore(df[f"{k}_Skoru_Ham"], f"{k}_Skoru")
    df["Final_Skor"] = [fr.calculate_final_score(m, s, r)
                        for m, s, r in zip(df.M_Skoru, df.S_Skoru, df.R_Skoru)]
    df["Adil_Skor"] = df.apply(fr.apply_fairness_boost, axis=1) if fr.fairness_enabled else df["Final_Skor"]
    dm, dr = (df["M_Skoru"] - old_m).abs().max(), (df["R_Skoru"] - old_r).abs().max()
    df = df.sort_values("Adil_Skor", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / args.output, index=False)

    # --- Ozet
    c = pd.Series(c_vals)
    print("\n=== OZET ===")
    print(f"Ortam: {platform.platform()}, Python {platform.python_version()}, cihaz {sc.device}")
    print(f"P_AI = {sc.human_score:.6f}; S_human = {h_vals[0]:.4f}")
    print(f"S_context: ort {c.mean():.4f}, std {c.std():.4f}, min {c.min():.4f}, max {c.max():.4f}")
    print(f"S ham: ort {pd.Series(s_vals).mean():.4f}, std {pd.Series(s_vals).std():.4f}, "
          f"min {min(s_vals):.4f}, max {max(s_vals):.4f}")
    print(f"M~ ve R~ eski CSV ile max fark (0 olmali): {dm:.2e}, {dr:.2e}")
    print(f"S hesaplama suresi / ilan: medyan {statistics.median(times) * 1000:.1f} ms, "
          f"ort {statistics.mean(times) * 1000:.1f} ms (n={len(times)})")
    print(f"Cikti: {args.output}")


if __name__ == "__main__":
    main()
