"""
Funções para avaliação de modelos de forecasting, incluindo backtesting com rolling origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dengue_forecast.models import Forecaster


@dataclass
class BacktestResult:
    model_name: str
    mae: float
    rmse: float
    smape: float


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def rolling_origin_splits(series: pd.Series, horizon: int, min_train_size: int):
    """Generator para cross-validation com origem rolante."""
    n = len(series)
    last_train_end = n - horizon
    for train_end in range(min_train_size, last_train_end + 1):
        train = series.iloc[:train_end]
        test = series.iloc[train_end : train_end + horizon]
        if len(test) == horizon:
            yield train, test


def run_backtest(
    series: pd.Series,
    model: Forecaster,
    horizon: int,
    min_train_size: int,
) -> tuple[dict | None, pd.DataFrame]:
    """Executa o backtesting para um único modelo e retorna as métricas e previsões."""
    print(f"  [INFO] Rodando backtest para: {model.name}")
    y_true_all = []
    y_pred_all = []
    rows = []

    for train, test in rolling_origin_splits(series, horizon=horizon, min_train_size=min_train_size):
        try:
            y_pred = model.forecast(train, horizon=len(test))
        except Exception as exc:
            print(f"    [WARN] Falha em {model.name}: {exc}")
            continue

        y_true = test.to_numpy(dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_pred) != len(y_true):
            print(f"    [WARN] Tamanho inválido em {model.name}: pred={len(y_pred)} true={len(y_true)}")
            continue

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        for dt, yt, yp in zip(test.index, y_true, y_pred):
            rows.append(
                {
                    "model": model.name,
                    "date": dt,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    if not y_true_all:
        return None, pd.DataFrame(rows)

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)

    metric_row = {
        "model": model.name,
        "mae": mae(y_true_cat, y_pred_cat),
        "rmse": rmse(y_true_cat, y_pred_cat),
        "smape": smape(y_true_cat, y_pred_cat),
        "n_predictions": int(len(y_true_cat)),
    }

    return metric_row, pd.DataFrame(rows)
