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


CITY_FILE_MAP = {
    "sao_paulo": "sao",
    "rio": "rio",
    "belo_horizonte": "belo",
    "brasilia": "brasilia",
    "fortaleza": "fortaleza",
    "recife": "recife",
    "manaus": "manaus",
    "salvador": "salvador",
}

CITY_DISPLAY = {
    "sao_paulo": "São Paulo",
    "rio": "Rio de Janeiro",
    "belo_horizonte": "Belo Horizonte",
    "brasilia": "Brasília",
    "fortaleza": "Fortaleza",
    "recife": "Recife",
    "manaus": "Manaus",
    "salvador": "Salvador",
}


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


def plot_all_cities_smape_heatmap() -> None:
    """Heatmap de sMAPE para todos os modelos e cidades."""
    frames = []
    for city_key, _ in CITY_FILE_MAP.items():
        path = Path(f"results/benchmark_{city_key}_metrics.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)[["model", "smape"]]
        df["city"] = CITY_DISPLAY.get(city_key, city_key)
        frames.append(df)

    if not frames:
        print("[WARN] Nenhum arquivo de métricas encontrado para heatmap.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    pivot = all_df.pivot(index="model", columns="city", values="smape")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "sMAPE (%)"},
    )
    ax.set_title("sMAPE por Modelo e Cidade — Benchmark Dengue 2010-2024", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out = FIGURES_DIR / "smape_heatmap_todas_cidades.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura salva em {out}")


def plot_timesfm_ranking() -> None:
    """Mostra o ranking do TimesFM em cada cidade e o sMAPE comparado ao melhor concorrente."""
    frames = []
    for city_key, _ in CITY_FILE_MAP.items():
        path = Path(f"results/benchmark_{city_key}_metrics.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path).sort_values("smape").reset_index(drop=True)
        df["rank"] = df.index + 1
        row_tfm = df[df["model"] == "timesfm"]
        if row_tfm.empty:
            continue
        rank = int(row_tfm["rank"].values[0])
        smape_tfm = float(row_tfm["smape"].values[0])
        best_other = df[df["model"] != "timesfm"].iloc[0]
        smape_best = float(best_other["smape"])
        frames.append({
            "city": CITY_DISPLAY.get(city_key, city_key),
            "rank": rank,
            "smape_timesfm": smape_tfm,
            "smape_best_other": smape_best,
            "best_other_model": best_other["model"],
        })

    if not frames:
        return

    df = pd.DataFrame(frames).sort_values("smape_timesfm")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Painel esquerdo: ranking por cidade
    colors = ["gold" if r == 1 else ("steelblue" if r <= 3 else "salmon") for r in df["rank"]]
    axes[0].barh(df["city"], df["rank"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Ranking (1 = melhor)")
    axes[0].set_title("Ranking do TimesFM por Cidade", fontweight="bold")
    axes[0].axvline(1, color="gold", linestyle="--", linewidth=1.5, label="1o lugar")
    axes[0].legend()
    for i, (rank, city) in enumerate(zip(df["rank"], df["city"])):
        axes[0].text(rank + 0.05, i, f"{rank}o", va="center", fontsize=10)

    # Painel direito: sMAPE TimesFM vs melhor concorrente
    x = range(len(df))
    width = 0.35
    bars1 = axes[1].bar([i - width/2 for i in x], df["smape_timesfm"], width, label="TimesFM", color="steelblue")
    bars2 = axes[1].bar([i + width/2 for i in x], df["smape_best_other"], width, label="Melhor concorrente", color="salmon")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df["city"], rotation=30, ha="right")
    axes[1].set_ylabel("sMAPE (%)")
    axes[1].set_title("TimesFM vs Melhor Concorrente por Cidade", fontweight="bold")
    axes[1].legend()
    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center", fontsize=8)

    plt.suptitle("Desempenho do TimesFM (Zero-Shot) — 8 Capitais Brasileiras", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "timesfm_ranking_todas_cidades.png"
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
    plot_all_cities_smape_heatmap()
    plot_timesfm_ranking()
