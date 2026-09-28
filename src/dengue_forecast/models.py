"""
Implementação dos modelos de forecasting usando skforecast.

Os modelos treinados aceitam log_target=True: ajuste em log1p(y) e previsão
devolvida na escala original por expm1. Isso permite que árvores e SARIMAX
acompanhem picos epidêmicos acima do máximo visto no treino.

Modelos incluídos:
- Seasonal naive (baseline de referência, y_{t+h} = y_{t+h-12})
- SARIMAX (baseline estatístico)
- Prophet (Meta)
- LightGBM
- XGBoost
- CatBoost
- RandomForest
- TimesFM (Google, zero-shot foundation model)
"""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# timesfm deve ser importado antes de catboost/lightgbm/xgboost para evitar
# conflito de bibliotecas nativas que causa segfault no macOS ARM
try:
    import timesfm as _timesfm
    _TIMESFM_AVAILABLE = True
except ImportError:
    _TIMESFM_AVAILABLE = False

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from skforecast.recursive import ForecasterRecursive
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

# Threads por estimador. Padrão 1: no macOS ARM, XGBoost multithread trava (0% CPU)
# depois que LightGBM/torch carregam outro libomp. Paralelize por cidade (make benchmark-all).
N_JOBS = int(os.environ.get("DENGUE_N_JOBS", "1"))

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from skforecast.exceptions import IgnoredArgumentWarning
    warnings.filterwarnings("ignore", category=IgnoredArgumentWarning)
except ImportError:
    pass


class Forecaster(ABC):
    """Interface para todos os modelos de forecasting."""

    name: str

    @abstractmethod
    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        """Recebe a série de treino e retorna a previsão para o horizonte."""
        pass


def _log_suffix(name: str, log_target: bool) -> str:
    return f"{name}_log" if log_target else name


class SeasonalNaiveForecaster(Forecaster):
    """Repete o valor do mesmo mês do ano anterior (baseline de referência)."""

    name = "seasonal_naive"

    def __init__(self, season: int = 12):
        self.season = season

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        last_season = train.to_numpy(dtype=float)[-self.season :]
        reps = int(np.ceil(horizon / self.season))
        return np.tile(last_season, reps)[:horizon]


class SarimaxForecaster(Forecaster):
    """Wrapper para o modelo SARIMAX de statsmodels."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), log_target: bool = False):
        self.order = order
        self.seasonal_order = seasonal_order
        self.log_target = log_target
        self.name = _log_suffix("sarimax", log_target)

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        y = np.log1p(train) if self.log_target else train
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.forecast(steps=horizon)
        result = np.asarray(pred, dtype=float)
        if self.log_target:
            # limita o expoente para evitar overflow quando o SARIMAX diverge
            result = np.expm1(np.clip(result, None, np.log1p(train.max()) + 5))
        return np.maximum(0, result)


class SkforecastWrapper(Forecaster):
    """Wrapper genérico para modelos de ML usando ForecasterRecursive do skforecast."""

    def __init__(self, estimator, lags: int = 24, name: str | None = None, log_target: bool = False):
        self.estimator = estimator
        self.lags = lags
        self.log_target = log_target
        base = name or estimator.__class__.__name__.lower().replace("regressor", "")
        self.name = _log_suffix(base, log_target)

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        transformer_y = (
            FunctionTransformer(np.log1p, np.expm1, check_inverse=False) if self.log_target else None
        )
        forecaster = ForecasterRecursive(
            estimator=self.estimator, lags=self.lags, transformer_y=transformer_y
        )
        forecaster.fit(y=train)
        predictions = forecaster.predict(steps=horizon)
        return np.maximum(0, predictions.to_numpy())


def lgbm_forecaster(lags: int = 24, log_target: bool = False) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=LGBMRegressor(random_state=42, verbosity=-1, n_jobs=N_JOBS),
        lags=lags,
        name="lgbm",
        log_target=log_target,
    )


def xgboost_forecaster(lags: int = 24, log_target: bool = False) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=N_JOBS),
        lags=lags,
        name="xgboost",
        log_target=log_target,
    )


def catboost_forecaster(lags: int = 24, log_target: bool = False) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=CatBoostRegressor(random_state=42, verbose=0, thread_count=N_JOBS, allow_writing_files=False),
        lags=lags,
        name="catboost",
        log_target=log_target,
    )


def randomforest_forecaster(lags: int = 24, log_target: bool = False) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=N_JOBS),
        lags=lags,
        name="randomforest",
        log_target=log_target,
    )


class ProphetForecaster(Forecaster):
    """Wrapper para o Prophet (Meta)."""

    def __init__(self, log_target: bool = False):
        self.log_target = log_target
        self.name = _log_suffix("prophet", log_target)

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        from prophet import Prophet  # lazy import

        y = np.log1p(train) if self.log_target else train
        df = y.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)

        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        preds = forecast["yhat"].iloc[-horizon:].to_numpy(dtype=float)
        if self.log_target:
            preds = np.expm1(np.clip(preds, None, np.log1p(train.max()) + 5))
        return np.maximum(0, preds)


class TimesFMForecaster(Forecaster):
    """Wrapper zero-shot para TimesFM 2.5 do Google.

    Modelo de fundação pré-treinado: não requer treinamento.
    O contexto histórico é passado diretamente ao modelo via HuggingFace.
    O modelo é carregado uma única vez e cacheado na classe.
    """

    name = "timesfm"
    _model = None  # cache de classe para evitar recarregamento a cada fold

    def _get_model(self):
        if TimesFMForecaster._model is None:
            if not _TIMESFM_AVAILABLE:
                raise ImportError("timesfm nao instalado. Execute: pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'")
            if N_JOBS > 0:
                import torch

                torch.set_num_threads(N_JOBS)
            model = _timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            model.compile(
                _timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=24,
                    normalize_inputs=True,
                    infer_is_positive=True,
                )
            )
            TimesFMForecaster._model = model
        return TimesFMForecaster._model

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        model = self._get_model()
        context = train.to_numpy(dtype=float)
        point_forecast, _ = model.forecast(horizon=horizon, inputs=[context])
        return np.maximum(0, point_forecast[0])
