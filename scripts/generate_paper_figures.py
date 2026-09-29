"""
Figuras do manuscrito (v2), geradas a partir das saídas de scripts/analyze_results.py.

Fig 1: séries mensais das 8 capitais (escala log) com a janela avaliada sombreada
Fig 2: forest plot da diferença pareada de sMAPE (TimesFM - comparador) com IC 95%
Fig 3: sMAPE por horizonte (1-12 meses), mediana entre cidades
Fig 4: heatmap de sMAPE (conjunto primário)

Exemplo:
  python scripts/generate_paper_figures.py --results-dir results/v2 --out-dir paper/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_results import CITY_LABELS, MODEL_LABELS, PRIMARY_MODELS  # noqa: E402

# Paleta de referência (dataviz): slot 1 azul, slot 2 laranja; demais em cinza recessivo
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#b5b3ab"
SEQ_CMAP = "Blues"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

DATA_FILES = {c: f"data/processed/dengue_monthly_{c}.csv" for c in CITY_LABELS}


def fig1_series(out: Path, first_test: pd.Timestamp) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 8.5), sharex=True)
    for ax, (city, label) in zip(axes.flat, CITY_LABELS.items()):
        s = pd.read_csv(DATA_FILES[city], index_col=0, parse_dates=True)["value"]
        ax.axvspan(first_test, s.index.max(), color=GRID, alpha=0.6, lw=0)
        ax.plot(s.index, s.values, color=BLUE, lw=1.2)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cases/month (log)")
    fig.text(0.5, -0.01, "Shaded: months covered by forecast windows (Jan 2014 - Dec 2024)", ha="center", color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig1_series.png")
    plt.close(fig)


def fig2_forest(out: Path, tests: pd.DataFrame) -> None:
    comps = [m for m in PRIMARY_MODELS if m != "timesfm"]
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.2), sharex=True, sharey=True)
    for ax, (city, label) in zip(axes.flat, CITY_LABELS.items()):
        t = tests[tests["city"] == city].set_index("comparator").loc[comps]
        y = np.arange(len(comps))[::-1]
        sig = t["p_holm"] < 0.05
        ax.axvline(0, color=INK2, lw=0.8)
        ax.hlines(y, t["diff_lo"], t["diff_hi"], color=INK2, lw=1.4)
        ax.scatter(t["diff_smape"], y, s=28, zorder=3, color=np.where(sig, BLUE, "white"), edgecolor=BLUE, lw=1.2)
        ax.set_yticks(y, [MODEL_LABELS[c] for c in comps])
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("TimesFM - comparator\nsMAPE (pp)")
    fig.text(
        0.5, -0.03,
        "Negative = TimesFM better. Bars: 95% paired block-bootstrap CI. Filled: Diebold-Mariano p<0.05 (Holm).",
        ha="center", color=INK2,
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_forest_timesfm_vs_comparators.png")
    plt.close(fig)


def fig3_horizon(out: Path, by_h: pd.DataFrame) -> None:
    med = by_h.groupby(["model", "step"])["smape"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    others_labeled = False
    for m in PRIMARY_MODELS:
        if m == "timesfm":
            kw = dict(color=BLUE, lw=2.2, zorder=4, label="TimesFM")
        elif m == "catboost_log":
            kw = dict(color=ORANGE, lw=2, zorder=3, label="CatBoost")
        elif m == "seasonal_naive":
            kw = dict(color=INK2, lw=1.6, ls="--", zorder=3, label="Seasonal naive")
        else:
            kw = dict(color=MUTED, lw=1.2, zorder=2)
            if not others_labeled:
                kw["label"] = "Other models (XGBoost, RF, LightGBM, SARIMA, Prophet)"
                others_labeled = True
        ax.plot(med.index, med[m], **kw)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Forecast horizon (months ahead)")
    ax.set_ylabel("sMAPE (%), median across 8 cities")
    ax.set_xlim(1, 12)
    fig.tight_layout()
    fig.savefig(out / "fig3_smape_by_horizon.png")
    plt.close(fig)


def fig4_heatmap(out: Path, metrics: pd.DataFrame) -> None:
    piv = metrics.pivot(index="model", columns="city", values="smape").loc[PRIMARY_MODELS, list(CITY_LABELS)]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(piv.values, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [CITY_LABELS[c] for c in piv.columns], rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]), [MODEL_LABELS[m] for m in piv.index])
    ax.grid(False)
    vmid = np.nanmean(piv.values)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            best = v == piv.iloc[:, j].min()
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > vmid else INK, fontweight="bold" if best else "normal")
    fig.colorbar(im, ax=ax, label="sMAPE (%)")
    fig.tight_layout()
    fig.savefig(out / "fig4_smape_heatmap.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/v2")
    ap.add_argument("--out-dir", default="paper/figures")
    args = ap.parse_args()
    rd, out = Path(args.results_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(rd / "metrics_primary.csv")
    tests = pd.read_csv(rd / "tests_primary.csv")
    by_h = pd.read_csv(rd / "by_horizon_primary.csv")

    fig1_series(out, pd.Timestamp("2014-01-01"))
    fig2_forest(out, tests)
    fig3_horizon(out, by_h)
    fig4_heatmap(out, metrics)
    print(f"[INFO] Figuras salvas em {out}")


if __name__ == "__main__":
    main()
