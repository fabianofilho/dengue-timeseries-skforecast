"""
Funções utilitárias para carregamento e preparação de dados de dengue.
"""
from __future__ import annotations

import pandas as pd


def load_monthly_series(
    csv_path: str,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Carrega uma série temporal mensal de dengue a partir de um arquivo CSV.

    Parameters
    ----------
    csv_path : str
        Caminho para o arquivo CSV.
    date_col : str
        Nome da coluna de data.
    value_col : str
        Nome da coluna de valor (casos).

    Returns
    -------
    pd.Series
        Série temporal com frequência mensal (MS).
    """
    df = pd.read_csv(csv_path, index_col=date_col, parse_dates=[date_col])
    if value_col not in df.columns:
        raise ValueError(f"Coluna '{value_col}' não encontrada em {csv_path}.")
    series = df[value_col].asfreq("MS")
    series.name = value_col
    return series
