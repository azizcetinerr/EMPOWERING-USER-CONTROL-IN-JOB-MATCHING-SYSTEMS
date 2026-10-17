"""Figure 2 of the paper: MMR lambda sweep of the main system, read from experiments/out/diagnostics.json.

  python -m experiments.plot_mmr      # writes WileyDesign/fairpass_v3/fig_mmr_lambda.pdf

Left panel: listings shared with the first ten of the main system without MMR.
Right panel: ILD@10 (1 - mean pairwise cosine of description embeddings of the first ten listings), with the
original run and BM25 as reference lines. One series per panel; no dual axis.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "experiments" / "out" / "diagnostics.json"
OUT = ROOT / "WileyDesign" / "fairpass_v3" / "fig_mmr_lambda.pdf"

SERIES = "#2a78d6"     # categorical slot 1 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(DIAG.read_text(encoding="utf-8"))
    main_label = d["main_system"]
    lambdas = ["off", "0.9", "0.7", "0.5", "0.2"]
    names = [main_label] + [f"{main_label} + MMR λ={l}" for l in lambdas[1:]]
    overlap = [d["top10_overlap_with_main"][n] for n in names]
    ild = [d["ILD@10"][n] for n in names]
    ref = {"original run": d["ILD@10"]["original_A (reported run)"], "BM25": d["ILD@10"]["BM25 (profile)"]}

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), constrained_layout=True)
    x = range(len(lambdas))
    for ax, y, title, ylim in [(axes[0], overlap, "(a) Listings kept from the first ten without MMR", (0, 11.8)),
                               (axes[1], ild, "(b) ILD@10", (0, 0.75))]:
        ax.plot(x, y, color=SERIES, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_xticks(list(x), lambdas)
        ax.set_xlabel("MMR λ")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for i in (0, len(y) - 1):
            ax.annotate(f"{y[i]:.3f}" if isinstance(y[i], float) else str(y[i]), (i, y[i]),
                        textcoords="offset points", xytext=(0, -12 if (ax is axes[1] and i == 0) else 7),
                        ha="center", fontsize=7, color=INK)
    for name, v in ref.items():
        axes[1].axhline(v, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        axes[1].text(1.02, v, f"{name}\n{v:.3f}", transform=axes[1].get_yaxis_transform(), ha="left",
                     va="center", fontsize=7, color=MUTED, clip_on=False)
    fig.savefig(OUT)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
