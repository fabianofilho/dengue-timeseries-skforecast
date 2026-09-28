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


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    """MASE com escala por fold (MAE in-sample do seasonal naive no treino daquele fold)."""
    return float(np.mean(np.abs(y_true - y_pred) / scale))


def seasonal_naive_scale(train: pd.Series, season: int = 12) -> float:
    """MAE in-sample do seasonal naive, denominador do MASE (Hyndman & Koehler, 2006)."""
    values = train.to_numpy(dtype=float)
    return float(np.mean(np.abs(values[season:] - values[:-season])))


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
    scale_all = []
    rows = []
    n_failed = 0

    for train, test in rolling_origin_splits(series, horizon=horizon, min_train_size=min_train_size):
        try:
            y_pred = model.forecast(train, horizon=len(test))
        except Exception as exc:
            print(f"    [WARN] Falha em {model.name}: {exc}")
            n_failed += 1
            continue

        y_true = test.to_numpy(dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_pred) != len(y_true):
            print(f"    [WARN] Tamanho inválido em {model.name}: pred={len(y_pred)} true={len(y_true)}")
            n_failed += 1
            continue

        scale = seasonal_naive_scale(train)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        scale_all.append(np.full(len(y_true), scale))

        # origin = primeiro mês previsto; o treino termina no mês anterior
        origin = test.index[0]
        for step, (dt, yt, yp) in enumerate(zip(test.index, y_true, y_pred), start=1):
            rows.append(
                {
                    "model": model.name,
                    "origin": origin,
                    "step": step,
                    "date": dt,
                    "y_true": yt,
                    "y_pred": yp,
                    "scale": scale,
                }
            )

    if not y_true_all:
        return None, pd.DataFrame(rows)

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    scale_cat = np.concatenate(scale_all)

    metric_row = {
        "model": model.name,
        "mae": mae(y_true_cat, y_pred_cat),
        "rmse": rmse(y_true_cat, y_pred_cat),
        "smape": smape(y_true_cat, y_pred_cat),
        "mase": mase(y_true_cat, y_pred_cat, scale_cat),
        "n_predictions": int(len(y_true_cat)),
        "n_failed_folds": n_failed,
    }

    return metric_row, pd.DataFrame(rows)
