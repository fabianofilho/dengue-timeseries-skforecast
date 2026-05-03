"""
Executa o benchmark de backtesting para múltiplos modelos de forecasting em um dataset de série temporal de dengue.

Exemplo de uso:
  python scripts/run_benchmark.py \
    --input-csv data/processed/dengue_monthly_sao_paulo.csv \
    --output-prefix results/benchmark_sao_paulo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dengue_forecast.evaluate import run_backtest
from dengue_forecast.models import (
    ProphetForecaster,
    SarimaxForecaster,
    TimesFMForecaster,
    catboost_forecaster,
    lgbm_forecaster,
    randomforest_forecaster,
    xgboost_forecaster,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark de modelos de forecasting de dengue")
    parser.add_argument("--input-csv", required=True, help="CSV com a série temporal mensal")
    parser.add_argument("--date-col", default="date", help="Nome da coluna de data")
    parser.add_argument("--value-col", default="value", help="Nome da coluna de valor (casos)")
    parser.add_argument("--horizon", type=int, default=12, help="Horizonte de previsão (meses)")
    parser.add_argument("--min-train-size", type=int, default=48, help="Janela mínima de treino (meses)")
    parser.add_argument(
        "--models",
        default="sarimax,prophet,lgbm,xgboost,catboost,randomforest,timesfm",
        help="Lista de modelos para rodar (separados por vírgula)",
    )
    parser.add_argument("--output-prefix", required=True, help="Prefixo para os arquivos de saída")
    parser.add_argument("--lags", type=int, default=24, help="Número de lags para modelos de ML")
    return parser.parse_args()


def build_models(model_names: list[str], lags: int):
    """Constrói a lista de objetos de modelo com base nos nomes fornecidos."""
    selected = {m.strip().lower() for m in model_names if m.strip()}
    models = []
    if "sarimax" in selected:
        models.append(SarimaxForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)))
    if "prophet" in selected:
        models.append(ProphetForecaster())
    if "lgbm" in selected:
        models.append(lgbm_forecaster(lags=lags))
    if "xgboost" in selected:
        models.append(xgboost_forecaster(lags=lags))
    if "catboost" in selected:
        models.append(catboost_forecaster(lags=lags))
    if "randomforest" in selected:
        models.append(randomforest_forecaster(lags=lags))
    if "timesfm" in selected:
        models.append(TimesFMForecaster())

    if not models:
        raise ValueError("Nenhum modelo válido selecionado.")
    return models


def main() -> None:
    args = parse_args()

    # Carrega e prepara a série temporal
    series = pd.read_csv(
        args.input_csv,
        index_col=args.date_col,
        parse_dates=[args.date_col],
    )[args.value_col].asfreq("MS")

    models = build_models(args.models.split(","), lags=args.lags)

    metrics_rows = []
    preds_frames = []

    print(f"[INFO] Iniciando benchmark para {args.input_csv}...")
    for model in models:
        metric_row, pred_df = run_backtest(
            series=series,
            model=model,
            horizon=args.horizon,
            min_train_size=args.min_train_size,
        )
        if metric_row is not None:
            metrics_rows.append(metric_row)
        if not pred_df.empty:
            preds_frames.append(pred_df)

    # Salva os resultados
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values(by="smape")
        metrics_path = output_prefix.with_name(output_prefix.name + "_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n[INFO] Métricas salvas em: {metrics_path}")
        print(metrics_df)

    if preds_frames:
        preds_df = pd.concat(preds_frames, ignore_index=True)
        preds_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")
        preds_df.to_csv(preds_path, index=False)
        print(f"[INFO] Previsões salvas em: {preds_path}")


if __name__ == "__main__":
    main()
