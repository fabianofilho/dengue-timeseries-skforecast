"""
Gera todas as métricas, intervalos de confiança, testes e tabelas do manuscrito a partir
das previsões do benchmark v2 (results/v2/benchmark_<cidade>_predictions.csv).

Nenhum número do paper deve ser digitado à mão: tudo sai deste script.

Inferência
- As 121 origens de previsão de cada cidade se sobrepõem (cada mês aparece em até 12 folds),
  então as 1.452 previsões NÃO são independentes. Um bootstrap iid sobre previsões
  subestimaria os ICs.
- ICs: moving block bootstrap sobre as origens (bloco de 12 origens, 2.000 réplicas),
  recalculando cada métrica agregada nas origens reamostradas. Diferenças pareadas
  (TimesFM - comparador) usam as mesmas réplicas.
- Teste de Diebold-Mariano sobre a série de perdas por origem (sMAPE médio dos 12 passos),
  variância de longo prazo Newey-West (Bartlett, lag 12) com correção de Harvey,
  Leybourne & Newbold (1997) e distribuição t(n-1). Holm sobre todas as comparações.

Exemplo:
  python scripts/analyze_results.py --results-dir results/v2 --tables-dir paper/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CITY_LABELS = {
    "sao": "Sao Paulo",
    "rio": "Rio de Janeiro",
    "belo": "Belo Horizonte",
    "brasilia": "Brasilia",
    "fortaleza": "Fortaleza",
    "recife": "Recife",
    "manaus": "Manaus",
    "salvador": "Salvador",
}

# Conjunto primário, definido antes de ver os resultados v2: modelos treinados em log1p(y),
# TimesFM na escala bruta (uso pretendido) e seasonal naive como piso de referência.
PRIMARY_MODELS = [
    "timesfm",
    "catboost_log",
    "xgboost_log",
    "randomforest_log",
    "lgbm_log",
    "sarimax_log",
    "prophet_log",
    "seasonal_naive",
]
# Sensibilidade: mesmos modelos ajustados na escala bruta (configuração da v1)
RAW_MODELS = ["timesfm", "catboost", "xgboost", "randomforest", "lgbm", "sarimax", "prophet", "seasonal_naive"]

MODEL_LABELS = {
    "timesfm": "TimesFM",
    "catboost_log": "CatBoost",
    "xgboost_log": "XGBoost",
    "randomforest_log": "Random Forest",
    "lgbm_log": "LightGBM",
    "sarimax_log": "SARIMA",
    "prophet_log": "Prophet",
    "seasonal_naive": "Seasonal naive",
    "catboost": "CatBoost (raw)",
    "xgboost": "XGBoost (raw)",
    "randomforest": "Random Forest (raw)",
    "lgbm": "LightGBM (raw)",
    "sarimax": "SARIMA (raw)",
    "prophet": "Prophet (raw)",
}

# Resultados publicados na v1 (manuscrito de 03/05/2026) para checar reprodutibilidade
V1_FILES = {
    "sao": "sao_paulo",
    "rio": "rio",
    "belo": "belo_horizonte",
    "brasilia": "brasilia",
    "fortaleza": "fortaleza",
    "recife": "recife",
    "manaus": "manaus",
    "salvador": "salvador",
}

N_BOOT = 2000
BLOCK = 12
HORIZON = 12
SEED = 20260928
EPS = 1e-8


def load_predictions(results_dir: Path) -> pd.DataFrame:
    frames = []
    for city in CITY_LABELS:
        path = results_dir / f"benchmark_{city}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["origin", "date"])
        df["city"] = city
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    err = df["y_true"] - df["y_pred"]
    df["abs_err"] = err.abs()
    df["sq_err"] = err**2
    df["smape_term"] = 200.0 * df["abs_err"] / (df["y_true"].abs() + df["y_pred"].abs() + EPS)
    df["mase_term"] = df["abs_err"] / df["scale"]
    return df


def origin_sums(df: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Somas por (modelo, origem): abs_err, sq_err, smape_term, mase_term, n. Shape (M, O, 5)."""
    cols = ["abs_err", "sq_err", "smape_term", "mase_term"]
    g = df[df["model"].isin(models)].groupby(["model", "origin"])[cols].agg(["sum"])
    g.columns = cols
    g["n"] = df[df["model"].isin(models)].groupby(["model", "origin"]).size()
    origins = sorted(df["origin"].unique())
    arr = np.full((len(models), len(origins), 5), np.nan)
    for i, m in enumerate(models):
        sub = g.loc[m].reindex(origins)
        arr[i] = sub[cols + ["n"]].to_numpy()
    if np.isnan(arr).any():
        missing = [models[i] for i in range(len(models)) if np.isnan(arr[i]).any()]
        raise ValueError(f"Origens faltando para {missing}; compare só origens comuns")
    return arr, origins


def metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    """s: (..., 5) já somado sobre origens."""
    n = s[..., 4]
    return {
        "smape": s[..., 2] / n,
        "mae": s[..., 0] / n,
        "rmse": np.sqrt(s[..., 1] / n),
        "mase": s[..., 3] / n,
    }


def block_bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Moving block bootstrap: (N_BOOT, n) índices de origem."""
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)
    return idx[:, :n]


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = HORIZON) -> tuple[float, float, float]:
    """DM com HAC Bartlett (lag h) + correção HLN. Retorna (diferença média, estatística, p bicaudal)."""
    d = loss_a - loss_b
    n = len(d)
    d_bar = d.mean()
    dc = d - d_bar
    lrv = np.dot(dc, dc) / n
    for k in range(1, h + 1):
        w = 1.0 - k / (h + 1)
        lrv += 2.0 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = d_bar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return float(d_bar), float(stat), float(p)


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def analyze_set(df: pd.DataFrame, models: list[str], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas + ICs + diferenças pareadas + DM (TimesFM vs cada comparador) por cidade."""
    metric_rows, test_rows = [], []
    for city in CITY_LABELS:
        sub = df[df["city"] == city]
        sums, origins = origin_sums(sub, models)
        point = metrics_from_sums(sums.sum(axis=1))
        idx = block_bootstrap_indices(len(origins), rng)
        boot = metrics_from_sums(sums[:, idx, :].sum(axis=2))  # (M, B)
        for i, m in enumerate(models):
            row = {"city": city, "model": m, "n_predictions": int(sums[i, :, 4].sum()), "n_origins": len(origins)}
            for k in ["smape", "mae", "rmse", "mase"]:
                row[k] = point[k][i]
                row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(boot[k][i], [2.5, 97.5])
            metric_rows.append(row)

        ref = models.index("timesfm")
        loss = sums[:, :, 2] / sums[:, :, 4]  # sMAPE médio por origem, (M, O)
        for j, m in enumerate(models):
            if j == ref:
                continue
            diff_boot = boot["smape"][ref] - boot["smape"][j]
            d_bar, stat, p = diebold_mariano(loss[ref], loss[j])
            test_rows.append(
                {
                    "city": city,
                    "comparator": m,
                    "smape_timesfm": point["smape"][ref],
                    "smape_comparator": point["smape"][j],
                    "diff_smape": point["smape"][ref] - point["smape"][j],
                    "diff_lo": np.percentile(diff_boot, 2.5),
                    "diff_hi": np.percentile(diff_boot, 97.5),
                    "dm_mean_origin_loss_diff": d_bar,
                    "dm_stat": stat,
                    "p_value": p,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    tests = pd.DataFrame(test_rows)
    tests["p_holm"] = holm(tests["p_value"].to_numpy())
    for k in ["smape", "mase"]:
        metrics[f"rank_{k}"] = metrics.groupby("city")[k].rank(method="min").astype(int)
    return metrics, tests


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def fmt_ci(row: pd.Series, k: str, nd: int = 1) -> str:
    return f"{fmt(row[k], nd)} ({fmt(row[k + '_lo'], nd)}-{fmt(row[k + '_hi'], nd)})"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def wide_table(metrics: pd.DataFrame, models: list[str], k: str, nd: int = 1, ci: bool = False) -> str:
    rows = []
    for city, label in CITY_LABELS.items():
        sub = metrics[metrics["city"] == city].set_index("model")
        best = sub.loc[models, k].idxmin()
        cells = []
        for m in models:
            txt = fmt_ci(sub.loc[m], k, nd) if ci else fmt(sub.loc[m, k], nd)
            cells.append(f"**{txt}**" if m == best else txt)
        rows.append([label] + cells)
    mean = metrics[metrics["model"].isin(models)].groupby("model")[k].mean()
    rows.append(["Mean (8 cities)"] + [fmt(mean[m], nd) for m in models])
    return md_table(["City"] + [MODEL_LABELS[m] for m in models], rows)


def reproducibility_check(results_dir: Path, v2: pd.DataFrame) -> pd.DataFrame:
    """Compara modelos em escala bruta da v2 com os CSVs de métricas publicados na v1."""
    rows = []
    for city, v1_name in V1_FILES.items():
        path = results_dir.parent / f"benchmark_{v1_name}_metrics.csv"
        if not path.exists():
            continue
        v1 = pd.read_csv(path).set_index("model")
        sub = v2[v2["city"] == city].set_index("model")
        for m in v1.index:
            if m in sub.index:
                rows.append(
                    {
                        "city": city,
                        "model": m,
                        "smape_v1": v1.loc[m, "smape"],
                        "smape_v2": sub.loc[m, "smape"],
                        "abs_diff": abs(v1.loc[m, "smape"] - sub.loc[m, "smape"]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/v2")
    ap.add_argument("--tables-dir", default="paper/tables")
    args = ap.parse_args()
    results_dir, tables_dir = Path(args.results_dir), Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_predictions(results_dir)

    # Primário
    metrics, tests = analyze_set(df, PRIMARY_MODELS, rng)
    metrics.to_csv(results_dir / "metrics_primary.csv", index=False)
    tests.to_csv(results_dir / "tests_primary.csv", index=False)

    # Sensibilidade 1: escala bruta (v1)
    metrics_raw, tests_raw = analyze_set(df, RAW_MODELS, rng)
    metrics_raw.to_csv(results_dir / "metrics_raw.csv", index=False)
    tests_raw.to_csv(results_dir / "tests_raw.csv", index=False)

    # Sensibilidade 2: sem a epidemia recorde de 2024 (janela de teste termina até dez/2023)
    last_origin = pd.Timestamp("2023-12-01") - pd.DateOffset(months=HORIZON - 1)
    df_pre = df[df["origin"] <= last_origin]
    metrics_pre, tests_pre = analyze_set(df_pre, PRIMARY_MODELS, rng)
    metrics_pre.to_csv(results_dir / "metrics_pre2024.csv", index=False)
    tests_pre.to_csv(results_dir / "tests_pre2024.csv", index=False)

    # Erro por horizonte (1..12)
    by_h = (
        df[df["model"].isin(PRIMARY_MODELS)]
        .groupby(["city", "model", "step"])[["smape_term", "mase_term"]]
        .mean()
        .rename(columns={"smape_term": "smape", "mase_term": "mase"})
        .reset_index()
    )
    by_h.to_csv(results_dir / "by_horizon_primary.csv", index=False)

    repro = reproducibility_check(results_dir, pd.concat([metrics_raw]))
    repro.to_csv(results_dir / "reproducibility_v1_v2.csv", index=False)

    # ---------------- Tabelas do manuscrito ----------------
    (tables_dir / "table2_smape_ci.md").write_text(
        "Table 2. sMAPE (%) with 95% block-bootstrap CI, primary model set\n\n"
        + wide_table(metrics, PRIMARY_MODELS, "smape", ci=True)
    )
    (tables_dir / "table3_mase.md").write_text(
        "Table 3. MASE (scaled by in-sample seasonal naive MAE), primary model set\n\n"
        + wide_table(metrics, PRIMARY_MODELS, "mase", nd=2)
    )
    (tables_dir / "tableS1_mae.md").write_text(
        "Table S1. MAE (cases/month), primary model set\n\n" + wide_table(metrics, PRIMARY_MODELS, "mae", nd=0)
    )
    (tables_dir / "tableS2_rmse.md").write_text(
        "Table S2. RMSE (cases/month), primary model set\n\n" + wide_table(metrics, PRIMARY_MODELS, "rmse", nd=0)
    )
    (tables_dir / "tableS3_smape_raw.md").write_text(
        "Table S3. sMAPE (%), models fitted on the raw scale (v1 configuration)\n\n"
        + wide_table(metrics_raw, RAW_MODELS, "smape")
    )
    (tables_dir / "tableS4_smape_pre2024.md").write_text(
        "Table S4. sMAPE (%), primary set, forecast windows ending by December 2023\n\n"
        + wide_table(metrics_pre, PRIMARY_MODELS, "smape")
    )

    test_rows = []
    for city, label in CITY_LABELS.items():
        for _, r in tests[tests["city"] == city].iterrows():
            test_rows.append(
                [
                    label,
                    MODEL_LABELS[r["comparator"]],
                    f"{r['diff_smape']:+.1f} ({r['diff_lo']:+.1f} to {r['diff_hi']:+.1f})",
                    f"{r['dm_stat']:.2f}",
                    f"{r['p_holm']:.3f}" if r["p_holm"] >= 0.001 else "<0.001",
                ]
            )
    (tables_dir / "tableS5_dm_tests.md").write_text(
        "Table S5. TimesFM minus comparator sMAPE (pp, 95% paired block-bootstrap CI) and "
        "Diebold-Mariano test (HLN-corrected, Holm-adjusted p)\n\n"
        + md_table(["City", "Comparator", "Difference (95% CI)", "DM", "p (Holm)"], test_rows)
    )

    # ---------------- Resumo para o texto ----------------
    def summarize(m: pd.DataFrame, t: pd.DataFrame, models: list[str]) -> dict:
        firsts = m[m["rank_smape"] == 1].groupby("model").size().reindex(models, fill_value=0)
        firsts_mase = m[m["rank_mase"] == 1].groupby("model").size().reindex(models, fill_value=0)
        best_comp = (
            t.loc[t.groupby("city")["smape_comparator"].idxmin()]
            .set_index("city")[["comparator", "smape_comparator", "diff_smape", "diff_lo", "diff_hi", "p_holm"]]
        )
        beats_naive = m.pivot(index="city", columns="model", values="smape")
        return {
            "mean_smape": m.groupby("model")["smape"].mean().round(1).reindex(models).to_dict(),
            "median_smape": m.groupby("model")["smape"].median().round(1).reindex(models).to_dict(),
            "mean_mase": m.groupby("model")["mase"].mean().round(2).reindex(models).to_dict(),
            "first_place_smape": firsts.to_dict(),
            "first_place_mase": firsts_mase.to_dict(),
            "timesfm_vs_best_comparator": best_comp.round(3).reset_index().to_dict(orient="records"),
            "n_cities_timesfm_sig_better_holm": int(((t["diff_smape"] < 0) & (t["p_holm"] < 0.05)).groupby(t["city"]).all().sum()),
            "n_tests_sig_better": int(((t["diff_smape"] < 0) & (t["p_holm"] < 0.05)).sum()),
            "n_tests_sig_worse": int(((t["diff_smape"] > 0) & (t["p_holm"] < 0.05)).sum()),
            "n_tests": int(len(t)),
            "cities_model_worse_than_naive": {
                mm: [c for c in beats_naive.index if beats_naive.loc[c, mm] > beats_naive.loc[c, "seasonal_naive"]]
                for mm in models
                if mm != "seasonal_naive"
            },
        }

    summary = {
        "primary": summarize(metrics, tests, PRIMARY_MODELS),
        "raw": summarize(metrics_raw, tests_raw, RAW_MODELS),
        "pre2024": summarize(metrics_pre, tests_pre, PRIMARY_MODELS),
        "n_origins_full": int(df["origin"].nunique()),
        "n_origins_pre2024": int(df_pre["origin"].nunique()),
        "first_origin": str(df["origin"].min().date()),
        "last_origin": str(df["origin"].max().date()),
        "reproducibility_max_abs_diff_smape": float(repro["abs_diff"].max()) if not repro.empty else None,
        "bootstrap": {"n_boot": N_BOOT, "block": BLOCK, "seed": SEED},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
