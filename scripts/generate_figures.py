"""
Gera figuras de análise exploratória e resultados do benchmark de dengue.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CITY_LABELS = {
    "sao": "São Paulo",
    "rio": "Rio de Janeiro",
    "belo": "Belo Horizonte",
    "brasilia": "Brasília",
    "fortaleza": "Fortaleza",
    "recife": "Recife",
    "manaus": "Manaus",
    "salvador": "Salvador",
}

sns.set_theme(style="whitegrid", palette="colorblind")


def plot_all_series() -> None:
    """Plota as séries temporais de todas as cidades."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=False)
    axes = axes.flatten()

    files = sorted(glob.glob("data/processed/dengue_monthly_*.csv"))
    for i, fpath in enumerate(files):
        city_key = Path(fpath).stem.replace("dengue_monthly_", "")
        city_name = CITY_LABELS.get(city_key, city_key.title())
        series = pd.read_csv(fpath, index_col="date", parse_dates=["date"])["value"]
        axes[i].plot(series.index, series.values, color="steelblue", linewidth=1)
        axes[i].set_title(city_name, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Ano")
        axes[i].set_ylabel("Casos mensais")
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("Séries Temporais de Casos de Dengue — Capitais Brasileiras (2010–2024)", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "series_temporais_dengue.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em {out}")


def plot_benchmark_results() -> None:
    """Plota os resultados do benchmark de São Paulo."""
    metrics_path = Path("results/benchmark_sao_paulo_metrics.csv")
    if not metrics_path.exists():
        print("[WARN] Arquivo de métricas não encontrado. Pule esta etapa.")
        return

    df = pd.read_csv(metrics_path)
    df = df.sort_values("smape")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_info = [
        ("mae", "MAE", "Mean Absolute Error"),
        ("rmse", "RMSE", "Root Mean Squared Error"),
        ("smape", "sMAPE (%)", "Symmetric MAPE"),
    ]

    for ax, (col, ylabel, title) in zip(axes, metrics_info):
        colors = ["gold" if i == 0 else "steelblue" for i in range(len(df))]
        bars = ax.barh(df["model"], df[col], color=colors)
        ax.set_xlabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}", va="center", fontsize=9)

    plt.suptitle("Benchmark de Modelos — São Paulo (2010–2024)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "benchmark_sao_paulo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em {out}")


def plot_predictions_vs_actual() -> None:
    """Plota as previsões vs. valores reais para São Paulo."""
    preds_path = Path("results/benchmark_sao_paulo_predictions.csv")
    if not preds_path.exists():
        print("[WARN] Arquivo de previsões não encontrado.")
        return

    df = pd.read_csv(preds_path, parse_dates=["date"])
    models = df["model"].unique()

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub = df[df["model"] == model_name].sort_values("date")
        ax.plot(sub["date"], sub["y_true"], label="Real", color="black", linewidth=1.5)
        ax.plot(sub["date"], sub["y_pred"], label="Previsto", color="steelblue", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(f"Modelo: {model_name.upper()}", fontweight="bold")
        ax.set_ylabel("Casos mensais")
        ax.legend()

    axes[-1].set_xlabel("Data")
    plt.suptitle("Previsões vs. Valores Reais — São Paulo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "predictions_vs_actual_sao_paulo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em {out}")


if __name__ == "__main__":
    plot_all_series()
    plot_benchmark_results()
    plot_predictions_vs_actual()
