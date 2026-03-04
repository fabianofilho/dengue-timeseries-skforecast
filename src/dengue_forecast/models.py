"""
Implementação dos modelos de forecasting usando skforecast.

Modelos incluídos:
- SARIMAX (baseline estatístico)
- LightGBM
- XGBoost
- CatBoost
- RandomForest
- Ridge
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)


class Forecaster(ABC):
    """Interface para todos os modelos de forecasting."""

    name: str

    @abstractmethod
    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        """Recebe a série de treino e retorna a previsão para o horizonte."""
        pass


class SarimaxForecaster(Forecaster):
    """Wrapper para o modelo SARIMAX de statsmodels."""

    name = "sarimax"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.forecast(steps=horizon)
        result = np.asarray(pred, dtype=float)
        return np.maximum(0, result)


class SkforecastWrapper(Forecaster):
    """Wrapper genérico para modelos de ML usando ForecasterRecursive do skforecast."""

    def __init__(self, estimator, lags: int = 24, name: str | None = None):
        self.estimator = estimator
        self.lags = lags
        self.name = name or estimator.__class__.__name__.lower().replace("regressor", "")

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        forecaster = ForecasterRecursive(estimator=self.estimator, lags=self.lags)
        forecaster.fit(y=train)
        predictions = forecaster.predict(steps=horizon)
        return np.maximum(0, predictions.to_numpy())


def lgbm_forecaster(lags: int = 24) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=LGBMRegressor(random_state=42, verbosity=-1),
        lags=lags,
        name="lgbm",
    )


def xgboost_forecaster(lags: int = 24) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=XGBRegressor(random_state=42, objective="reg:squarederror"),
        lags=lags,
        name="xgboost",
    )


def catboost_forecaster(lags: int = 24) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=CatBoostRegressor(random_state=42, verbose=0),
        lags=lags,
        name="catboost",
    )


def randomforest_forecaster(lags: int = 24) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=RandomForestRegressor(random_state=42, n_estimators=100),
        lags=lags,
        name="randomforest",
    )


def ridge_forecaster(lags: int = 24) -> SkforecastWrapper:
    return SkforecastWrapper(
        estimator=Ridge(random_state=42),
        lags=lags,
        name="ridge",
    )
