# A Zero-Shot Foundation Model Versus City-Trained Models for 12-Month Dengue Forecasting in Eight Brazilian State Capitals: A Rolling-Origin Benchmark

**Running title:** Zero-shot versus trained models for dengue forecasting in Brazil

**Authors:** Fabiano Novaes Barcellos Filho^1^

**Affiliations:**  
^1^ Laboratorio de Big Data e Analise Preditiva em Saude (LABDAPS), Faculdade de Saude Publica, Universidade de Sao Paulo, Sao Paulo, Brazil

**Corresponding author:** Fabiano Novaes Barcellos Filho, fabiano.nb@gmail.com

**Keywords:** dengue; forecasting; foundation model; time series; machine learning; Brazil; TimesFM; epidemiological surveillance

**Word count (Introduction to Discussion):** approximately 3,210

---

## Abstract

**Background.** Time-series foundation models can forecast a new series without being trained on it, which would spare surveillance teams from building and maintaining one model per municipality. Whether they forecast dengue as well as models trained on each city's own history remains unclear, and the answer depends on how fairly the baselines are configured.

**Methods.** We compared TimesFM 2.5, used zero-shot, with a seasonal naive benchmark, SARIMA, Prophet, and four tree ensembles (CatBoost, XGBoost, Random Forest, LightGBM) fitted on log-transformed counts, using monthly notified dengue cases from eight Brazilian state capitals (January 2010 to December 2024, InfoDengue). An expanding-window rolling-origin design produced 121 forecast origins per city with a 12-month horizon (1,452 forecasts per model and city). The primary metric was sMAPE; MASE was secondary. Uncertainty was quantified with a moving block bootstrap over forecast origins and Diebold-Mariano tests with Holm correction.

**Results.** Mean sMAPE across cities was 71.8% for CatBoost, 73.3% for Random Forest, 75.3% for TimesFM, and 79.3% for the seasonal naive forecast. TimesFM had the lowest sMAPE in 2 of 8 cities and the lowest MASE in 6 of 8. Most paired differences had confidence intervals that included zero, and only 1 of 56 TimesFM comparisons remained significant after Holm correction. TimesFM was the least accurate model in Belo Horizonte (sMAPE 97.1% vs 62.0% for CatBoost; difference 35.1 points, 95% CI 19.9 to 56.9). TimesFM was most accurate one month ahead, while at 12 months no model beat the seasonal naive forecast. When the trained models were fitted on raw counts instead, the tree ensembles fell behind the seasonal naive forecast in 6 of 8 cities, which made TimesFM appear superior.

**Conclusions.** Without any city-specific training, TimesFM forecast monthly dengue cases about as accurately as the best models trained on each city, but it was not reliably better, and it failed in one of eight cities. Comparisons of forecasting methods for dengue need a seasonal naive benchmark, fair scaling of the trained baselines, and uncertainty estimates that account for overlapping forecast windows.

## Author summary

Health authorities in Brazil plan dengue control, hospital beds, and supply purchases months ahead, and forecasts of case counts can support that planning. Building a separate forecasting model for each city takes time and expertise. New "foundation" forecasting models, trained by technology companies on millions of unrelated time series, can produce a forecast for any series without further training. We tested whether one of them, TimesFM, forecasts monthly dengue cases in eight Brazilian capitals as well as conventional models fitted to each city's data. Over forecasts made from 2014 to 2024 as if in real time, TimesFM was about as accurate as the best conventional models, with no clear winner, and it performed poorly in Belo Horizonte. A simple rule that repeats last year's count for the same month was hard to beat at 12 months ahead. We also show that fitting the conventional models on untransformed counts makes the foundation model look better than it is. Foundation models are a reasonable low-maintenance option for dengue forecasting, but they should be checked city by city and compared against simple benchmarks before use.

---

## 1. Introduction

Dengue is the most common arboviral infection worldwide, with an estimated 390 million infections per year [Bhatt2013]. Brazil carries a large share of this burden: in 2024 it reported more than 6 million probable cases, the highest annual count on record, with outbreaks in all five macroregions [GurgelGoncalves2024]. The costs include medical care, lost productivity, and deaths [Siqueira2022].

Forecasts of dengue cases months ahead can help health authorities position supplies, schedule vector control, and plan hospital capacity before peaks [Roster2022]. In Brazil, the InfoDengue system (Fiocruz/FGV) publishes surveillance data for every municipality and has supported several forecasting studies [Codeco2018infodengue]. Most published approaches, however, rely on models fitted separately for each location, from ARIMA-family models to tree ensembles and neural networks, which must be refitted and monitored as local dynamics change [Leung2023, Roster2022, Chen2025lstm].

Time-series foundation models offer another route. They are pre-trained on large and heterogeneous collections of time series and can forecast a new series directly from its history, with no fitting [Das2024timesfm, Ansari2024chronos, Woo2024moirai]. TimesFM, a decoder-only transformer from Google Research, performed competitively with supervised baselines on public benchmarks in zero-shot mode [Das2024timesfm], and its 2.5 checkpoint was released in 2025 [TimesFM25card]. If such a model forecast dengue as well as city-specific models, a national surveillance platform could cover every municipality with a single model.

Benchmarks of this question are easy to tilt, though. A foundation model applies its own normalization, while trained baselines may be fitted on raw counts, a setting in which tree ensembles cannot forecast above the largest value seen in training. Comparisons may also omit a seasonal naive benchmark or treat forecasts from overlapping windows as independent observations, which makes confidence intervals too narrow [Hewamalage2023].

We compared TimesFM 2.5, used zero-shot, with a seasonal naive benchmark and six models fitted on each city's history for 12-month-ahead forecasting of monthly dengue cases in eight Brazilian state capitals. All trained models were fitted on log-transformed counts, and uncertainty was estimated with methods that respect the dependence between overlapping forecast windows. Our objective was to determine whether a zero-shot foundation model can match or exceed city-trained models in this setting.

---

## 2. Methods

This study follows the TRIPOD+AI 2024 reporting guideline for predictive model evaluation [Collins2024tripod]. Code, processed data, and all result files are available at https://github.com/fabianofilho/dengue-timeseries-skforecast (release tag `paper-v2`).

### 2.1 Data source and study population

We used dengue case counts retrieved from InfoDengue, a surveillance platform maintained by Fundacao Oswaldo Cruz (Fiocruz) and Fundacao Getulio Vargas (FGV) that aggregates mandatory notifications from Brazil's national notifiable diseases information system (SINAN) [Codeco2018infodengue]. Data were obtained through the InfoDengue public API for eight state capitals: Sao Paulo (SP), Rio de Janeiro (RJ), Belo Horizonte (MG), Brasilia (DF), Fortaleza (CE), Recife (PE), Manaus (AM), and Salvador (BA), covering the five geographic macroregions of Brazil.

The API returns weekly counts by epidemiological week. We used the notified case count (`casos`) rather than the nowcast estimate (`casos_est`). These are the consolidated counts available when the data were downloaded (March 2026), not the provisional counts that would have been available at each forecast origin (see Limitations). Weekly counts were summed into calendar months, assigning each epidemiological week to the month in which it starts. The API returned 782 consecutive weeks per city, from the week starting 3 January 2010 to the week starting 22 December 2024, so December 2024 lacks the epidemiological week starting on 29 December. The study period spanned January 2010 to December 2024 (180 months per city). No week was missing within this range and no month had zero notified cases, so no imputation was needed. Only anonymized, publicly available aggregate data were used.

### 2.2 Outcome

The outcome was the total number of notified dengue cases per city per calendar month. Because dengue counts vary over three orders of magnitude between inter-epidemic troughs and epidemic peaks, all trained models (SARIMA, Prophet, and the four tree ensembles) were fitted on log1p(y) and their forecasts back-transformed with expm1. This choice was fixed before the analysis reported here was run. Tree ensembles fitted on the raw scale cannot predict values above the maximum seen in training, which disadvantages them in record epidemic years such as 2024; the raw-scale versions are reported as a sensitivity analysis. TimesFM received the raw series, which is its intended input (it applies its own internal normalization). Negative forecasts were clipped to zero.

### 2.3 Forecasting models

Eight models formed the primary comparison set.

**Seasonal naive.** The forecast for each month repeats the count observed in the same calendar month of the last year of the training window. It requires no fitting and is the reference that any forecasting model should beat [Hewamalage2023].

**SARIMA.** Seasonal ARIMA with order (1,1,1) and seasonal order (1,1,0)12, fitted by maximum likelihood in statsmodels. No exogenous regressors were used. Orders were fixed across cities and folds.

**Prophet.** An additive decomposition model with piecewise linear trend, yearly Fourier seasonality, and automatic changepoint detection [Taylor2018prophet]. Weekly and daily components were disabled given the monthly frequency. Default priors were used.

**LightGBM, XGBoost, CatBoost, Random Forest.** Four tree ensembles [Ke2017lightgbm, Chen2016xgboost, Prokhorenkova2018catboost, Breiman2001] were wrapped in a recursive multi-step forecaster (ForecasterRecursive, skforecast) with 24 autoregressive lags and no covariates. Library default hyperparameters were used (Random Forest with 100 trees), with random_state = 42 and a single thread per model for deterministic results. Each model was refitted at every forecast origin on all data available up to that origin.

**TimesFM 2.5.** A decoder-only transformer foundation model released by Google Research in September 2025 as the checkpoint google/timesfm-2.5-200m-pytorch [Das2024timesfm, TimesFM25card]. According to its model card, it was pre-trained on the GIFT-Eval pre-training corpus [Aksu2024gifteval], Wikimedia pageviews (to November 2023), Google Trends top queries (to end of 2022), and synthetic data. We used it zero-shot: no fine-tuning and no city-specific adaptation. At each origin the full available history (48 to 168 months, below the configured maximum context of 512) was passed as context, with input normalization and the positivity constraint enabled. The point forecast was used.

### 2.4 Evaluation design

We used rolling-origin evaluation with an expanding window [Tashman2000, Hewamalage2023]. The first origin used 48 months of training data (January 2010 to December 2013) and each subsequent origin added one month. At each origin, every model produced forecasts for the next 12 months. With T = 180 months, minimum training size m = 48, and horizon h = 12, there were T - m - h + 1 = 121 origins per city (forecast windows starting January 2014 to January 2024), giving 1,452 forecasts per model per city. Every model saw exactly the same training data and was scored on exactly the same target months. No information from a forecast window was used for fitting, feature construction, or model configuration at that origin [Kapoor2023].

### 2.5 Performance metrics

Metrics were computed by pooling all 1,452 forecasts per model and city.

**sMAPE** (primary): sMAPE = (100/n) * sum[ |y_t - yhat_t| / ((|y_t| + |yhat_t|) / 2) ]. sMAPE is scale-free, which allows comparison across cities whose counts differ by two orders of magnitude, and it is common in the dengue forecasting literature. It is not symmetric in practice: it penalizes under-forecasts of low counts more than over-forecasts [Hewamalage2023]. We therefore also report MASE.

**MASE**: absolute error divided by the in-sample mean absolute error of the seasonal naive forecast on the training window of the same origin [HyndmanKoehler2006]. MASE below 1 means the model beat the in-sample seasonal naive benchmark.

**MAE and RMSE** in cases per month are reported in the supplement; RMSE is dominated by the largest epidemic months.

### 2.6 Statistical inference

The 121 forecast origins of a city overlap (each target month is forecast from up to 12 origins), and forecast errors are serially correlated, so the 1,452 forecasts are not independent. Treating them as independent would produce confidence intervals that are too narrow. We therefore resampled forecast origins rather than individual forecasts, using a moving block bootstrap [Kunsch1989] with blocks of 12 consecutive origins and 2,000 replicates, and recomputed each pooled metric in every replicate. For the difference between TimesFM and each comparator, both models were evaluated on the same resampled origins, giving a paired 95% percentile interval.

We also tested TimesFM against each comparator with the Diebold-Mariano test [DieboldMariano1995] applied to the series of 121 origin-level losses (mean sMAPE over the 12 forecast months of each origin). The long-run variance was estimated with a Bartlett (Newey-West) kernel with 12 lags, and the statistic was multiplied by the Harvey-Leybourne-Newbold small-sample correction and compared with a t distribution with 120 degrees of freedom [HarveyLeybourneNewbold1997]. P-values were adjusted with the Holm procedure across all 56 comparisons (8 cities x 7 comparators) [Holm1979].

### 2.7 Sensitivity analyses

Two sensitivity analyses were pre-specified. First, all trained models were refitted on the untransformed counts, to show how the choice of scale affects the comparison. Second, because 2024 was the largest dengue epidemic on record in Brazil [GurgelGoncalves2024], we restricted the evaluation to the 109 origins whose 12-month forecast window ended by December 2023.

### 2.8 Software and reproducibility

Analyses ran in Python 3.10.18 with pandas 2.3.1, numpy 2.0.2, scikit-learn 1.6.1, skforecast 0.20.1, statsmodels 0.14.4, prophet 1.3.0, lightgbm 4.6.0, xgboost 3.0.2, catboost 1.2.8, scipy 1.15.3, torch 2.8.0, and the timesfm package 2.0.0, on an Apple M4 CPU (TimesFM inference on CPU). Every number in the tables and text is produced by `scripts/analyze_results.py` from the saved forecasts; the full pipeline runs with `make benchmark-all analysis figures`.

---

## 3. Results

### 3.1 Dengue series

Table 1 summarizes the eight series. Mean monthly counts ranged from 649 in Salvador to 9,509 in Sao Paulo, and every series was strongly right-skewed: the mean exceeded the median in all cities, and the coefficient of variation ranged from 108% to 431%. The largest monthly count, 331,402 cases in Sao Paulo, occurred during the 2024 epidemic. Figure 1 shows the series on a log scale.

**Table 1. Monthly notified dengue cases by city, January 2010 to December 2024**

| City | Region | Months | Mean | Median | Max | Min | CV (%) | Total |
|---|---|---|---|---|---|---|---|---|
| Sao Paulo | Southeast | 180 | 9,509 | 1,198.5 | 331,402 | 255 | 431 | 1,711,689 |
| Belo Horizonte | Southeast | 180 | 6,034 | 988.5 | 122,453 | 148 | 264 | 1,086,158 |
| Brasilia | Central-West | 180 | 4,025 | 1,134.5 | 98,704 | 118 | 282 | 724,545 |
| Rio de Janeiro | Southeast | 180 | 3,547 | 635.5 | 65,260 | 28 | 246 | 638,433 |
| Fortaleza | Northeast | 180 | 2,122 | 1,018.0 | 18,986 | 133 | 144 | 382,040 |
| Recife | Northeast | 180 | 1,031 | 487.0 | 8,504 | 67 | 135 | 185,593 |
| Manaus | North | 180 | 735 | 264.5 | 21,605 | 69 | 286 | 132,282 |
| Salvador | Northeast | 180 | 649 | 364.5 | 4,331 | 19 | 108 | 116,817 |

*CV: coefficient of variation (SD/mean x 100). Source: InfoDengue [Codeco2018infodengue].*

### 3.2 Primary comparison

Table 2 shows sMAPE with 95% block-bootstrap confidence intervals for the eight models of the primary set. Averaged across cities, CatBoost had the lowest sMAPE (71.8%), followed by Random Forest (73.3%), TimesFM (75.3%), XGBoost (78.7%), the seasonal naive forecast (79.3%), Prophet (80.1%), LightGBM (81.6%), and SARIMA (83.5%). The best model differed by city: CatBoost ranked first in three cities (Sao Paulo, Belo Horizonte, Recife), TimesFM in two (Fortaleza, Salvador), and Random Forest, Prophet, and the seasonal naive forecast in one each. TimesFM ranked second in Rio de Janeiro, Brasilia, and Recife, fifth in Sao Paulo and Manaus, and last in Belo Horizonte.

The confidence intervals were wide and overlapped for most models within each city. In Sao Paulo, for example, sMAPE was 75.6% (95% CI 58.4 to 87.3) for CatBoost and 78.0% (61.1 to 87.3) for TimesFM.

**Table 2. sMAPE (%) with 95% block-bootstrap CI, primary model set**

| City | TimesFM | CatBoost | XGBoost | Random Forest | LightGBM | SARIMA | Prophet | Seasonal naive |
|---|---|---|---|---|---|---|---|---|
| Sao Paulo | 78.0 (61.1-87.3) | **75.6 (58.4-87.3)** | 81.2 (68.3-90.5) | 77.0 (60.9-87.0) | 88.9 (81.8-93.8) | 87.8 (74.3-100.2) | 77.3 (60.0-87.8) | 77.6 (63.8-85.1) |
| Rio de Janeiro | 96.8 (84.8-109.6) | 97.5 (77.6-113.7) | 101.5 (89.0-112.8) | **93.5 (75.9-105.9)** | 101.2 (84.8-112.3) | 101.7 (87.8-111.9) | 109.6 (85.2-131.1) | 104.1 (89.3-119.0) |
| Belo Horizonte | 97.1 (85.7-111.0) | **62.0 (45.4-75.8)** | 76.4 (58.0-90.8) | 70.3 (52.0-85.8) | 92.1 (81.5-102.5) | 95.8 (87.8-107.7) | 80.7 (68.4-90.9) | 79.6 (69.6-89.5) |
| Brasilia | 74.3 (62.8-86.4) | 76.6 (67.1-84.0) | 90.3 (76.5-101.1) | 81.5 (68.8-90.9) | 84.2 (74.5-90.1) | 80.2 (70.7-89.3) | 79.3 (63.8-92.8) | **74.2 (64.5-83.9)** |
| Fortaleza | **64.2 (56.6-75.0)** | 69.4 (58.6-86.6) | 79.7 (69.1-93.5) | 68.4 (56.8-84.0) | 73.7 (66.5-85.7) | 73.4 (65.3-83.8) | 69.7 (58.9-86.6) | 69.6 (59.5-86.8) |
| Recife | 67.3 (56.7-78.1) | **65.9 (50.3-80.5)** | 73.4 (58.7-86.7) | 71.0 (57.4-82.7) | 77.5 (65.6-88.9) | 87.1 (75.1-96.6) | 81.5 (61.5-103.4) | 83.3 (67.6-101.0) |
| Manaus | 56.0 (47.3-61.9) | 55.7 (47.6-61.6) | 54.8 (44.3-61.6) | 53.9 (45.5-59.4) | 62.8 (54.7-69.3) | 62.4 (50.3-71.5) | **53.8 (41.3-59.4)** | 59.1 (46.7-68.4) |
| Salvador | **68.9 (59.5-82.3)** | 71.8 (56.7-86.5) | 72.2 (57.6-84.0) | 71.0 (57.2-84.9) | 72.3 (63.6-82.1) | 79.5 (62.6-97.4) | 88.9 (68.4-108.0) | 87.2 (64.7-112.3) |
| Mean (8 cities) | 75.3 | 71.8 | 78.7 | 73.3 | 81.6 | 83.5 | 80.1 | 79.3 |

*Values are sMAPE (%) with 95% moving block bootstrap intervals over 121 forecast origins. Bold: lowest sMAPE in the city. Trained models fitted on log1p(y); TimesFM zero-shot on raw counts.*

By MASE (Table 3), TimesFM had the lowest error in six of eight cities and the lowest mean across cities (1.33, vs 1.37 for CatBoost and 1.39 for Random Forest), although the differences between the leading models were small. MASE exceeded 1 for every model in Sao Paulo, Brasilia, and Salvador, largely because the in-sample seasonal naive error used for scaling was small relative to the errors during the 2024 epidemic (see Section 3.5). The seasonal naive forecast itself had a mean MASE of 1.53. TimesFM also had the lowest MAE in five cities (Table S1); CatBoost had the lowest MAE in Sao Paulo and Belo Horizonte and Random Forest in Manaus. RMSE results are shown in Table S2.

**Table 3. MASE (scaled by in-sample seasonal naive MAE), primary model set**

| City | TimesFM | CatBoost | XGBoost | Random Forest | LightGBM | SARIMA | Prophet | Seasonal naive |
|---|---|---|---|---|---|---|---|---|
| Sao Paulo | **4.31** | 4.33 | 4.35 | 4.33 | 4.48 | 5.42 | 4.36 | 4.43 |
| Rio de Janeiro | **0.48** | 0.53 | 0.67 | 0.51 | 0.60 | 0.94 | 0.51 | 0.53 |
| Belo Horizonte | 0.92 | **0.84** | 0.95 | 0.90 | 1.06 | 2.20 | 0.93 | 1.18 |
| Brasilia | **2.42** | 2.53 | 2.76 | 2.62 | 2.61 | 3.74 | 2.66 | 2.76 |
| Fortaleza | **0.67** | 0.74 | 0.97 | 0.74 | 0.85 | 1.06 | 0.75 | 0.79 |
| Recife | **0.59** | 0.66 | 0.72 | 0.72 | 0.77 | 1.71 | 0.80 | 0.86 |
| Manaus | 0.18 | 0.18 | 0.19 | 0.17 | 0.20 | 0.21 | **0.17** | 0.20 |
| Salvador | **1.07** | 1.13 | 1.14 | 1.11 | 1.14 | 2.04 | 1.41 | 1.51 |
| Mean (8 cities) | 1.33 | 1.37 | 1.47 | 1.39 | 1.47 | 2.16 | 1.45 | 1.53 |

*MASE below 1 indicates lower error than the in-sample seasonal naive forecast. Bold: lowest MASE in the city.*

### 3.3 Paired comparisons with TimesFM

Figure 2 and Table S5 show the paired difference in sMAPE between TimesFM and each comparator. Of the 56 comparisons, 13 had a 95% bootstrap interval that excluded zero. Eight favored TimesFM (against XGBoost in Brasilia and Fortaleza; LightGBM, SARIMA, Prophet, and the seasonal naive forecast in Recife; and Prophet and the seasonal naive forecast in Salvador) and five favored the comparator, all in Belo Horizonte. After Holm correction of the Diebold-Mariano tests, a single comparison remained significant: TimesFM was more accurate than SARIMA in Recife (difference -19.9 points, 95% CI -26.9 to -11.2). Against CatBoost, the model with the lowest mean sMAPE, the difference ranged from -5.2 points in Fortaleza to +2.4 points in Sao Paulo in seven cities, with every interval including zero. In Belo Horizonte, TimesFM was 35.1 points worse (95% CI 19.9 to 56.9; Holm-adjusted p = 0.055).

### 3.4 Error by forecast horizon

Error rose with the horizon for all trained models and for TimesFM (Figure 3). One month ahead, TimesFM had the lowest median sMAPE across cities (39.9%, vs 42.0% for CatBoost, 42.4% for Random Forest, and 76.9% for the seasonal naive forecast). The advantage disappeared by six months (80.9% for TimesFM vs 77.0% for CatBoost). At 12 months, the seasonal naive forecast had the lowest median sMAPE (79.1%), below CatBoost (80.3%), TimesFM (84.7%), and Random Forest (86.2%).

### 3.5 Sensitivity analyses

When the trained models were fitted on raw counts (Table S3), their mean sMAPE increased to between 85.6% (CatBoost) and 115.4% (Prophet), and the tree ensembles did worse than the seasonal naive forecast in six of eight cities. In that configuration TimesFM ranked first in five cities and the seasonal naive forecast in the other three, and no trained model ranked first anywhere. As a reproducibility check, these raw-scale results matched an independent earlier run of the same pipeline (files in `results/` of the repository) to within 0.46 sMAPE points for SARIMA in Recife and within 0.0001 points for every other model.

Restricting the evaluation to the 109 origins whose forecast windows ended by December 2023, before the 2024 epidemic (Table S4), left the four leading models in the same order: mean sMAPE was 70.1% for CatBoost, 71.5% for Random Forest, 74.0% for TimesFM, and 78.2% for the seasonal naive forecast. TimesFM remained the least accurate model in Belo Horizonte, where its difference from CatBoost was the only comparison favoring a comparator that survived Holm correction (38.4 points, 95% CI 20.1 to 59.5). Mean MASE fell below 1 for all models except SARIMA, confirming that the high MASE values in the full analysis were driven by 2024.

---

## 4. Discussion

### 4.1 Main findings

Used without any training on dengue data, TimesFM 2.5 forecast monthly dengue cases in eight Brazilian capitals about as well as tree ensembles fitted to each city's history. It had the lowest scaled error (MASE) in six cities and the lowest error one month ahead, but its mean sMAPE was slightly higher than that of CatBoost and Random Forest, and almost none of the differences between the leading models could be distinguished from sampling variability. It also failed badly in Belo Horizonte. The most accurate description of the evidence is that the zero-shot model was comparable to, not better than, the best city-trained models. Because the study was not designed with an equivalence margin, the wide intervals do not establish equivalence either.

### 4.2 How a benchmark can overstate a foundation model

Our sensitivity analyses show how easily this comparison can be tilted in favor of the foundation model. Had we fitted the trained models on raw counts and omitted the seasonal naive benchmark, TimesFM would have ranked first in seven of eight cities (Table S3). Three design choices prevent this. First, fitting the trained models on log-transformed counts, which suits counts spanning three orders of magnitude, improved their mean sMAPE by about 14 points for CatBoost, 16 for Random Forest, and 22 to 35 for SARIMA and Prophet. Tree ensembles fitted on raw counts cannot forecast above the largest value in their training data, which handicaps them in epidemic years. Second, the seasonal naive benchmark revealed that the raw-scale tree ensembles were worse than repeating last year's counts in six of eight cities, a failure that a comparison among trained models alone would not show. Third, resampling forecast origins in blocks, rather than treating the 1,452 overlapping forecasts as independent, produced confidence intervals 3.4 to 4.9 times wider than a naive bootstrap over individual forecasts (median 4.5), and those naive intervals would have made small differences look reliable. Each of these is a known pitfall in forecast evaluation [Hewamalage2023]. We suggest that comparisons of foundation models in epidemiology report a seasonal naive benchmark, give trained baselines a scale treatment comparable to the normalization the foundation model applies internally, and use uncertainty estimates that account for overlapping forecast windows.

### 4.3 Belo Horizonte

TimesFM's failure in Belo Horizonte was consistent across the full and pre-2024 analyses. The city's series alternates sharp epidemic years with long troughs, and the tree ensembles, which learn from lagged values of the same city, captured this pattern better than a model relying only on general patterns learned elsewhere. We could not determine why TimesFM underperformed there, and eight cities are too few to identify which series characteristics predict failure. The practical point is that a zero-shot model's accuracy should be checked for each location before it replaces a locally validated model.

### 4.4 Implications for dengue surveillance

The main operational appeal of a foundation model is that it needs no fitting, tuning, or retraining. Our results suggest that this convenience comes at little cost in average accuracy for monthly dengue counts in large Brazilian cities, and that TimesFM is a reasonable default for short-range forecasts, where it performed best. At 12 months, however, no method beat the seasonal naive forecast, which implies that year-ahead point forecasts of monthly counts carry little information beyond the seasonal pattern. For annual planning, probabilistic forecasts and scenario ranges are likely more useful than point forecasts, and forecasts with climate covariates may add information that univariate models cannot [Barcellos2024, Fang2024].

### 4.5 Limitations

This study has several limitations. First, we forecast notified cases, which reflect reporting and testing practices as well as transmission and underestimate the number of infections [Bhatt2013]. We also used consolidated counts rather than the provisional counts available in real time; because recent weeks are revised upward as late notifications arrive, all models would face larger errors in operational use, and nowcasting would be needed before forecasting [Codeco2018infodengue]. Second, weekly counts were assigned to calendar months by the start date of each epidemiological week, which adds noise to monthly totals. Third, no model used climate or other covariates, and trained models used fixed, untuned hyperparameters; tuned or covariate-augmented models might outperform all models evaluated here. Fourth, we evaluated one foundation model checkpoint and only its point forecasts; other foundation models [Ansari2024chronos, Woo2024moirai], fine-tuning, and TimesFM's quantile forecasts were not assessed. Fifth, TimesFM's pre-training corpus includes epidemiological series from the United States (CDC influenza-like illness and Project Tycho notifiable diseases) [Aksu2024gifteval] and Wikipedia and Google Trends series that overlap our evaluation period in time [TimesFM25card]. We found no InfoDengue or Brazilian dengue series in the documented corpus, so direct leakage is unlikely, but indirect information cannot be ruled out. Sixth, sMAPE penalizes under-forecasts of small counts more than over-forecasts; we therefore also report MASE, whose conclusions agreed in direction. Finally, the analysis covered eight large capitals, and results may differ for smaller municipalities with sparse counts.

### 4.6 Conclusion

A zero-shot foundation model forecast monthly dengue cases in eight Brazilian capitals about as accurately as the best models trained on each city, without any local fitting, but it was not reliably better and it failed in one city. At a 12-month horizon, no model outperformed repeating last year's counts. Foundation models are a reasonable low-maintenance option for dengue surveillance, provided they are validated locally and compared against simple benchmarks and fairly configured trained models.

---

## Acknowledgements

Surveillance data were obtained from the InfoDengue platform (Fiocruz/FGV). TimesFM was developed by Google Research and is distributed through Hugging Face.

## Data availability statement

Monthly dengue case data are publicly available via the InfoDengue API (https://info.dengue.mat.br/api/). All analysis code, processed data, forecasts, and result files are available at https://github.com/fabianofilho/dengue-timeseries-skforecast (release tag `paper-v2`).

## Conflict of interest

The author declares no conflict of interest.

## Ethics statement

This study used publicly available, aggregated surveillance data without individual-level records. No ethics committee review was required.

---

## References

1. Bhatt S et al. The global distribution and burden of dengue. Nature. 2013;496:504-507. doi:10.1038/nature12060 [Bhatt2013]
2. Leung XY et al. A systematic review of dengue outbreak prediction models. PLOS NTD. 2023. doi:10.1371/journal.pntd.0010631 [Leung2023]
3. GurgelGoncalves R et al. The greatest Dengue epidemic in Brazil. Rev Soc Bras Med Trop. 2024. doi:10.1590/0037-8682-0113-2024 [GurgelGoncalves2024]
4. Siqueira Junior JB et al. Epidemiology and costs of dengue in Brazil: a systematic literature review. Int J Infect Dis. 2022. doi:10.1016/j.ijid.2022.06.050 [Siqueira2022]
5. Roster K et al. Machine-Learning-Based Forecasting of Dengue Fever in Brazilian Cities. Am J Epidemiol. 2022. doi:10.1093/aje/kwac090 [Roster2022]
6. Codeco C et al. Infodengue: A nowcasting system for the surveillance of arboviruses in Brazil. Rev Epidemiol Sante Publique. 2018. doi:10.1016/j.respe.2018.05.408 [Codeco2018infodengue]
7. Fang L et al. Meteorological factors cannot be ignored in ML-based methods for predicting dengue. Int J Biometeorol. 2024. doi:10.1007/s00484-023-02605-1 [Fang2024]
8. Das A et al. A decoder-only foundation model for time-series forecasting. ICML 2024. arXiv:2310.10688 [Das2024timesfm]
9. Collins GS et al. TRIPOD+AI statement. BMJ. 2024. doi:10.1136/bmj-2023-078378 [Collins2024tripod]
10. Taylor SJ, Letham B. Forecasting at Scale. Am Stat. 2018. doi:10.1080/00031305.2017.1380080 [Taylor2018prophet]
11. Tashman LJ. Out-of-sample tests of forecasting accuracy. Int J Forecasting. 2000. doi:10.1016/S0169-2070(00)00065-0 [Tashman2000]
12. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023. doi:10.1016/j.patter.2023.100804 [Kapoor2023]
13. Hewamalage H et al. Forecast evaluation for data scientists: common pitfalls and best practices. DMKD. 2023. doi:10.1007/s10618-022-00894-5 [Hewamalage2023]
14. Chen X, Moraga P. Forecasting dengue across Brazil with LSTM neural networks. BMC Public Health. 2025. doi:10.1186/s12889-025-22106-7 [Chen2025lstm]
15. Barcellos C et al. Climate change, thermal anomalies, and the recent progression of dengue in Brazil. Sci Rep. 2024. doi:10.1038/s41598-024-56044-y [Barcellos2024]
16. Ansari AF et al. Chronos: Learning the Language of Time Series. TMLR. 2024. arXiv:2403.07815 [Ansari2024chronos]
17. Woo G et al. Unified Training of Universal Time Series Forecasting Transformers (Moirai). ICML 2024. arXiv:2402.02592 [Woo2024moirai]
18. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016. doi:10.1145/2939672.2939785 [Chen2016xgboost]
19. Ke G et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017. [Ke2017lightgbm]
20. Prokhorenkova L et al. CatBoost: Unbiased Boosting with Categorical Features. NeurIPS 2018. arXiv:1706.09516 [Prokhorenkova2018catboost]
21. Breiman L. Random Forests. Machine Learning. 2001. doi:10.1023/A:1010933404324 [Breiman2001]
22. Hyndman RJ, Koehler AB. Another look at measures of forecast accuracy. Int J Forecasting. 2006. doi:10.1016/j.ijforecast.2006.03.001 [HyndmanKoehler2006]
23. Diebold FX, Mariano RS. Comparing Predictive Accuracy. J Bus Econ Stat. 1995. doi:10.1080/07350015.1995.10524599 [DieboldMariano1995]
24. Harvey D, Leybourne S, Newbold P. Testing the equality of prediction mean squared errors. Int J Forecasting. 1997. doi:10.1016/S0169-2070(96)00719-4 [HarveyLeybourneNewbold1997]
25. Kunsch HR. The Jackknife and the Bootstrap for General Stationary Observations. Ann Stat. 1989. doi:10.1214/aos/1176347265 [Kunsch1989]
26. Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70. [Holm1979]
27. Aksu T et al. GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. 2024. arXiv:2410.10393 [Aksu2024gifteval]
28. Google Research. TimesFM 2.5 (200M) model card. Hugging Face, 2025. https://huggingface.co/google/timesfm-2.5-200m-pytorch [TimesFM25card]

*Full BibTeX available in paper/refs/references.bib*

---

## Figures

**Figure 1. Monthly notified dengue cases in eight Brazilian state capitals, January 2010 to December 2024.** Log scale. The shaded area marks the months covered by forecast windows (January 2014 to December 2024).

**Figure 2. Paired difference in sMAPE between TimesFM and each comparator, by city.** Negative values favor TimesFM. Points are point estimates; bars are 95% paired moving block bootstrap intervals over forecast origins. Filled points indicate Diebold-Mariano tests significant at 0.05 after Holm correction across the 56 comparisons.

**Figure 3. sMAPE by forecast horizon.** Median across the eight cities of the sMAPE at each horizon from 1 to 12 months, primary model set. TimesFM, CatBoost, and the seasonal naive forecast are highlighted.

**Figure 4. sMAPE by model and city (primary model set).** Darker cells indicate higher error; the lowest value in each city is in bold.

## Supporting information

**Table S1.** MAE (cases per month), primary model set.  
**Table S2.** RMSE (cases per month), primary model set.  
**Table S3.** sMAPE for all models fitted on untransformed counts.  
**Table S4.** sMAPE for forecast windows ending by December 2023.  
**Table S5.** Paired sMAPE differences (TimesFM minus comparator) with 95% block-bootstrap intervals and Holm-adjusted Diebold-Mariano p-values.  
All supporting tables are generated by `scripts/analyze_results.py` and stored in `paper/tables/`.
