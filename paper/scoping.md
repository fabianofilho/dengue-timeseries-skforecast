# Scoping — Dengue Forecasting Paper

## Research question (PIRD format)

**Population:** Brazilian state capitals with dengue endemicity  
**Index model:** TimesFM 2.5 (Google, zero-shot foundation model)  
**Reference models:** SARIMAX, Prophet, LightGBM, XGBoost, CatBoost, RandomForest  
**Design:** Rolling-origin backtesting, 12-month horizon, 2010-2024  
**Outcome:** Forecast accuracy — sMAPE, MAE, RMSE

**Full question:** Do zero-shot foundation time-series models (TimesFM) outperform
traditional statistical and machine learning models in 12-month-ahead forecasting
of monthly dengue cases across eight Brazilian state capitals (2010-2024)?

## Study type

Comparative predictive modeling study.  
Reporting guideline: **TRIPOD+AI 2024**

## Journal target

Primary: PLOS Neglected Tropical Diseases  
Backup: Epidemics / BMC Infectious Diseases

## Data

- Source: InfoDengue (Fiocruz/FGV) — public, aggregated surveillance data
- Period: January 2010 to December 2024 (180 monthly observations per city)
- Cities: Sao Paulo, Rio de Janeiro, Belo Horizonte, Brasilia, Fortaleza, Recife, Manaus, Salvador
- Ethics: public aggregated data, no individual records — ethics waiver applicable

## Authors

- Fabiano Bozza Filho (corresponding) — [affiliation TBD]

## Key results (from benchmark)

| City           | TimesFM sMAPE | TimesFM rank | Best competitor      | Competitor sMAPE |
|----------------|--------------|-------------|----------------------|-----------------|
| Manaus         | 56.0%        | 1st         | XGBoost              | 65.5%           |
| Fortaleza      | 64.2%        | 1st         | CatBoost             | 77.5%           |
| Recife         | 67.3%        | 1st         | RandomForest         | 81.2%           |
| Salvador       | 68.9%        | 1st         | XGBoost              | 72.1%           |
| Brasilia       | 74.3%        | 1st         | CatBoost             | 76.6%           |
| Sao Paulo      | 78.0%        | 1st         | CatBoost             | 78.7%           |
| Rio de Janeiro | 96.8%        | 1st         | XGBoost              | 119.7%          |
| Belo Horizonte | 97.1%        | 4th         | XGBoost              | 88.7%           |

- TimesFM 1st in 7/8 cities
- Mean sMAPE TimesFM: 75.3% vs CatBoost (best ML): 80.1%
- n_predictions per model per city: 1,452 (121 folds x 12 steps)

## Timeline

- Manuscript draft: current session
- References: pending literature search
- Submission: TBD
