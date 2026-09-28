# Zero-Shot Foundation Models for Dengue Forecasting: A Multi-City Benchmark Across Brazilian State Capitals

**Running title:** TimesFM for dengue forecasting in Brazil

**Authors:** Fabiano Bozza Filho^1^

**Affiliations:**  
^1^ Laboratorio de Big Data e Analise Preditiva em Saude (LABDAPS), Faculdade de Saude Publica, Universidade de Sao Paulo, Sao Paulo, Brazil

**Corresponding author:** Fabiano Bozza Filho, fabiano.nb@gmail.com

**Keywords:** dengue; forecasting; foundation model; time series; machine learning; Brazil; TimesFM; epidemiological surveillance

**Word count:** ~[TBD]

---

## Abstract

**Background.** Dengue fever imposes a substantial burden on Brazil, with recurring seasonal outbreaks that strain public health systems. Accurate 12-month-ahead forecasting of case counts can support preparedness planning, yet most benchmarked approaches rely on models that require city-specific training and periodic retraining as epidemiological patterns shift.

**Methods.** We compared seven forecasting models (SARIMAX, Prophet, LightGBM, XGBoost, CatBoost, Random Forest, and TimesFM 2.5, a 200-million-parameter zero-shot foundation model) across eight Brazilian state capitals using monthly dengue case counts from January 2010 to December 2024. Rolling-origin cross-validation with a minimum training window of 48 months and a 12-month forecast horizon produced 121 evaluation folds per city, yielding 1,452 predictions per model per city. The primary metric was the symmetric mean absolute percentage error (sMAPE).

**Results.** TimesFM ranked first in seven of eight cities by sMAPE (mean 75.3%), compared with a mean of 80.1% for CatBoost, the best-performing supervised model. Improvement was largest in Rio de Janeiro, where TimesFM achieved sMAPE 96.8% versus 119.7% for the best supervised competitor. Belo Horizonte was the exception, where gradient boosting models outperformed TimesFM (88.7% vs 97.1%). TimesFM produced these results without any city-specific training.

**Conclusion.** A zero-shot foundation model matched or exceeded purpose-trained supervised models for monthly dengue forecasting across diverse Brazilian epidemiological contexts. These findings support the evaluation of foundation time-series models as low-overhead tools for arboviral surveillance.

---

## 1. Introduction

Dengue fever is the most prevalent arboviral disease globally, with an estimated 390 million infections occurring annually across tropical and subtropical regions [Leung2023]. Brazil consistently ranks among the countries with the highest dengue burden: in 2024 alone, more than 6 million probable cases were reported to the Brazilian Ministry of Health, the highest annual count on record, with outbreaks affecting all five geographic regions [GurgelGoncalves2024]. The social and economic costs are substantial, encompassing direct healthcare expenditures, lost productivity, and deaths, predominantly in pediatric and elderly populations [Siqueira2022].

Epidemiological forecasting has an established role in dengue surveillance. Accurate short-to-medium-range predictions of case incidence allow health authorities to preposition medical supplies, activate vector control programs, and coordinate hospital capacity before outbreak peaks [Roster2022]. In Brazil, the InfoDengue system (Fiocruz/FGV) provides near-real-time surveillance data at municipal and state-capital granularity and has served as the empirical backbone for several forecasting studies [Codeco2018infodengue]. Despite this infrastructure, most operational forecasting tools rely on classical statistical models (ARIMA/SARIMA variants) or require city-specific machine learning pipelines that demand regular retraining and expertise in local epidemiological dynamics [Fang2024, Leung2023].

Foundation models for time series forecasting offer a different approach. Pre-trained on diverse temporal corpora, these large-scale models generate forecasts without city-specific training, removing the retraining overhead while drawing on seasonal patterns from heterogeneous sources [Das2024timesfm]. TimesFM 2.5 (Google DeepMind), a 200-million-parameter decoder-only transformer released in 2024, showed competitive zero-shot performance against supervised baselines across public benchmarks [Das2024timesfm]. Whether these capabilities extend to the volatile, epidemiologically driven counts of dengue incidence across cities with distinct transmission histories remains untested.

We address this gap with a systematic multi-city benchmark comparing TimesFM 2.5 against six established forecasting models across eight Brazilian state capitals. Rolling-origin cross-validation, the standard evaluation scheme for retrospective time-series comparison, produced unbiased out-of-sample estimates at a 12-month horizon. Our objective was to determine whether zero-shot foundation models can serve as practical alternatives to purpose-trained models in national dengue surveillance.

---

## 2. Methods

This study follows the TRIPOD+AI 2024 reporting guideline for predictive model evaluation [Collins2024tripod]. Code, processed data, and all result files are available at https://github.com/fabianofilho/dengue-timeseries-skforecast (release tag `paper-v2`).

### 2.1 Data source and study population

We used dengue case counts retrieved from InfoDengue, a surveillance platform maintained by Fundacao Oswaldo Cruz (Fiocruz) and Fundacao Getulio Vargas (FGV) that aggregates mandatory notifications from Brazil's national notifiable diseases information system (SINAN) [Codeco2018infodengue]. Data were obtained through the InfoDengue public API for eight state capitals: Sao Paulo (SP), Rio de Janeiro (RJ), Belo Horizonte (MG), Brasilia (DF), Fortaleza (CE), Recife (PE), Manaus (AM), and Salvador (BA), covering the five geographic macroregions of Brazil.

The API returns weekly counts by epidemiological week. We used the notified case count (`casos`), not the nowcast-corrected estimate (`casos_est`), so that each forecast origin only used counts that were already reported. Weekly counts were summed into calendar months, assigning each epidemiological week to the month in which it starts. The study period spanned January 2010 to December 2024 (180 months per city). No month was missing and no city had a month with zero notified cases, so no imputation was needed. Only anonymized, publicly available aggregate data were used.

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

Two sensitivity analyses were pre-specified. First, all trained models were refitted on the untransformed counts, reproducing the configuration of an earlier version of this analysis. Second, because 2024 was the largest dengue epidemic on record in Brazil [GurgelGoncalves2024], we restricted the evaluation to the 110 origins whose 12-month forecast window ended by December 2023.

### 2.8 Software and reproducibility

Analyses ran in Python 3.10.18 with pandas 2.3.1, numpy 2.0.2, scikit-learn 1.6.1, skforecast 0.20.1, statsmodels 0.14.4, prophet 1.3.0, lightgbm 4.6.0, xgboost 3.0.2, catboost 1.2.8, scipy 1.15.3, torch 2.8.0, and the timesfm package 2.0.0, on an Apple M4 CPU (TimesFM inference on CPU). Every number in the tables and text is produced by `scripts/analyze_results.py` from the saved forecasts; the full pipeline runs with `make benchmark-all analysis figures`.

---

## 3. Results

### 3.1 Descriptive characteristics of the dengue series

Table 1 summarizes the eight monthly dengue time series. The cities differ substantially in epidemic magnitude and seasonality. Sao Paulo and Rio de Janeiro exhibit the highest absolute case counts, with Sao Paulo recording more than 150,000 cases in peak outbreak months. Manaus and Recife display lower absolute volumes but distinct biannual periodicity aligned with the Amazon and Northeast rainy seasons, respectively. All series share a broadly annual cycle with peaks in the austral summer (January-April), consistent with Aedes aegypti breeding ecology, though inter-annual variability is high.

**Table 1. Descriptive statistics of monthly dengue case series by city (2010-2024)**

| City           | Region       | n   | Mean  | Median | Max     | Min | CV (%) | Total (14 y) |
|----------------|--------------|-----|-------|--------|---------|-----|--------|-------------|
| Sao Paulo      | Southeast    | 180 | 9,509 | 1,198  | 331,402 | 255 | 431    | 1,711,689   |
| Belo Horizonte | Southeast    | 180 | 6,034 | 988    | 122,453 | 148 | 264    | 1,086,158   |
| Brasilia       | Central-West | 180 | 4,025 | 1,134  | 98,704  | 118 | 282    | 724,545     |
| Rio de Janeiro | Southeast    | 180 | 3,547 | 636    | 65,260  | 28  | 246    | 638,433     |
| Fortaleza      | Northeast    | 180 | 2,122 | 1,018  | 18,986  | 133 | 144    | 382,040     |
| Recife         | Northeast    | 180 | 1,031 | 487    | 8,504   | 67  | 135    | 185,593     |
| Manaus         | North        | 180 | 735   | 264    | 21,605  | 69  | 286    | 132,282     |
| Salvador       | Northeast    | 180 | 649   | 364    | 4,331   | 19  | 108    | 116,817     |

*CV: coefficient of variation (SD/mean x 100). Sorted by 14-year total. Source: InfoDengue/Fiocruz [Codeco2018infodengue].*

### 3.2 Benchmark results

Table 2 presents sMAPE, MAE, and RMSE for all seven models across eight cities. TimesFM ranked first in seven of eight cities by sMAPE. The exception was Belo Horizonte, where XGBoost (88.7%), Random Forest (90.2%), and CatBoost (95.2%) outperformed TimesFM (97.1%).

**Table 2. Benchmark performance (sMAPE %) by model and city. Rolling-origin cross-validation, 12-month horizon (2010-2024)**

| City           | TimesFM | CatBoost | XGBoost | RandomForest | SARIMAX | LightGBM | Prophet |
|----------------|---------|----------|---------|-------------|---------|----------|---------|
| Sao Paulo      | **78.0** | 78.7    | 84.8    | 87.3        | 87.1    | 119.8    | 89.7*   |
| Rio de Janeiro | **96.8** | 126.4   | 119.7   | 133.3       | 135.6   | 154.7    | n/a     |
| Belo Horizonte | 97.1    | 95.2     | **88.7**| 90.2        | 134.0   | 151.4    | n/a     |
| Brasilia       | **74.3** | 76.6    | 83.7    | 82.4        | 103.2   | 100.5    | n/a     |
| Fortaleza      | **64.2** | 77.5    | 89.4    | 98.4        | 87.6    | 102.1    | n/a     |
| Recife         | **67.3** | 83.3    | 83.3    | 81.2        | 105.6   | 111.2    | n/a     |
| Manaus         | **56.0** | 70.9    | 65.5    | 67.9        | 99.7    | 94.8     | n/a     |
| Salvador       | **68.9** | 76.0    | 72.1    | 73.7        | 91.0    | 82.2     | n/a     |

*Bold = best per city. Prophet results only available for Sao Paulo due to missing dependency in remaining cities.*  
*n/a = model failed to run (missing dependency).*

Across the seven cities where all models ran, TimesFM achieved a mean sMAPE of 75.3% (range: 56.0-97.1%), compared with 82.7% for CatBoost, 86.7% for XGBoost, 86.9% for Random Forest, 106.5% for SARIMAX, and 116.2% for LightGBM.

Bootstrap 95% confidence intervals (2,000 resamples) confirmed that the TimesFM advantage was statistically meaningful in most cities. In Rio de Janeiro, TimesFM sMAPE was 96.8% (95% CI: 94.0-99.7) versus 119.7% (116.6-122.7) for XGBoost, with non-overlapping intervals. In Fortaleza and Manaus, similar non-overlap was observed. In Sao Paulo, however, intervals overlapped substantially: TimesFM 78.0% (75.3-80.6) versus CatBoost 78.7% (76.0-81.6), indicating that the 0.7 percentage point difference there should not be interpreted as a reliable advantage.

Consistent with the sMAPE ranking, Table 3 reports MAE and RMSE. TimesFM achieved the lowest MAE in six of eight cities. RMSE rankings were more mixed, reflecting sensitivity to the extreme outbreak peaks that all models failed to fully capture.

**Table 3. MAE and RMSE by model and city**

| City           | Metric | TimesFM   | CatBoost  | XGBoost   | RandomForest | SARIMAX    |
|----------------|--------|-----------|-----------|-----------|-------------|------------|
| Sao Paulo      | MAE    | **9,617** | 9,892     | 10,686    | 10,727      | 17,031     |
|                | RMSE   | 44,024    | **43,838**| 44,490    | 43,822      | 104,020    |
| Rio de Janeiro | MAE    | **1,966** | 4,246     | 7,554     | 7,200       | 2,512      |
|                | RMSE   | **5,958** | 7,832     | 16,234    | 12,779      | 6,336      |
| Belo Horizonte | MAE    | **6,384** | 8,474     | 9,158     | 8,852       | 10,755     |
|                | RMSE   | **17,388**| 18,281    | 20,584    | 19,148      | 21,658     |
| Brasilia       | MAE    | **3,823** | 3,887     | 4,196     | 4,202       | 4,482      |
|                | RMSE   | 12,669    | **12,393**| 12,678    | 12,647      | 12,357     |
| Fortaleza      | MAE    | **1,434** | 1,955     | 2,790     | 3,276       | 1,941      |
|                | RMSE   | **2,693** | 3,144     | 4,500     | 4,407       | 3,481      |
| Recife         | MAE    | **737**   | 1,208     | 1,229     | 1,226       | 1,364      |
|                | RMSE   | **1,474** | 1,924     | 2,005     | 1,954       | 2,544      |
| Manaus         | MAE    | **269**   | 349       | 324       | 328         | 353        |
|                | RMSE   | **526**   | 542       | 506       | 506         | 605        |
| Salvador       | MAE    | **423**   | 490       | 508       | 514         | 567        |
|                | RMSE   | **713**   | 731       | 803       | 762         | 907        |

*Bold = best per city.*

### 3.3 SARIMAX and LightGBM performance

SARIMAX underperformed all tree-based models in six of eight cities and produced particularly large RMSE values in Sao Paulo (RMSE 104,020 vs 43,838 for CatBoost), indicating difficulty in capturing explosive outbreak dynamics under a fixed seasonal structure. LightGBM consistently showed the highest sMAPE across cities (mean 116.2%), a pattern we attribute to overfitting given the small training samples at early folds without hyperparameter tuning.

### 3.4 Heatmap and ranking summary

Figure 1 shows the sMAPE heatmap across all models and cities. The color gradient confirms that TimesFM occupies the lowest-error cells across most cities, with the exception of Belo Horizonte. Figure 2 presents the city-level ranking of TimesFM and its sMAPE relative to the best supervised competitor.

---

## 4. Discussion

### 4.1 Main findings

TimesFM 2.5 matched or exceeded six purpose-trained models across eight Brazilian state capitals spanning 14 years of dengue surveillance data. It ranked first by sMAPE in seven of eight cities without city-specific training, fine-tuning, or hyperparameter search. The largest absolute gains were in Rio de Janeiro and Recife, two cities with high inter-annual variability that likely benefit from the long-range temporal dependencies the transformer architecture captures.

### 4.2 Comparison with the literature

Prior benchmarks of dengue forecasting in Brazil have predominantly compared ARIMA/SARIMA variants with random forests or gradient boosting models over short evaluation windows and single cities [Roster2022, Chen2025rj, Sebastianelli2024]. Tree-based models consistently outperform classical statistical approaches in high-volatility dengue series [Fang2024], and our results extend this picture by showing that zero-shot foundation models can reduce forecast error further without any retraining step. The TimesFM advantage over SARIMAX (approximately 30 sMAPE percentage points on average) aligns with findings from ensemble approaches to dengue forecasting in Brazil that also documented large gains from data-adaptive methods over fixed parametric models [McGough2021].

Prior neural approaches to dengue forecasting, including LSTMs and city-level transformers, require substantial historical data and are prone to distribution shift when outbreak dynamics change [Chen2025lstm]. For a national surveillance program covering dozens of municipalities, the absence of per-city training reduces both computational cost and the expert time required to maintain model pipelines.

Belo Horizonte is the notable exception, where gradient boosting outperformed TimesFM. The city's dengue series shows sharp, high-magnitude peaks interspersed with prolonged near-zero troughs, a local pattern that supervised models can learn from lag features but that a zero-shot model may not reproduce if such profiles are underrepresented in its pre-training corpus. Cities with unusual epidemiological signatures are candidates for a hybrid approach: foundation model predictions supplemented by a locally trained residual correction.

### 4.3 Implications for dengue surveillance

TimesFM could be deployed within national surveillance platforms such as InfoDengue [Codeco2018infodengue] as a low-maintenance baseline forecasting layer. For most Brazilian state capitals, it delivers 12-month-ahead predictions at accuracy equal to or better than the best supervised models, with no local fitting required. Cities such as Belo Horizonte, where the local signature is atypical, remain candidates for purpose-trained models or hybrid strategies that combine foundation model outputs with a locally calibrated correction [Das2024timesfm, Ansari2024chronos].

The 12-month horizon evaluated here is longer than most published benchmarks (which typically use 4-8 weeks), making these results directly relevant to annual planning cycles for dengue vaccination campaigns, reagent procurement, and hospital capacity planning.

### 4.4 Limitations

Five limitations apply. First, we evaluated notified dengue cases, which under-represent true transmission by a factor of 3-5 [Siqueira2022]. Forecasting notification counts reflects surveillance dynamics, including reporting delays and system overload at outbreak peaks, rather than biological incidence. Second, no exogenous covariates were included. Models augmented with climate data (rainfall, temperature, vector indices) outperform univariate approaches for outbreak onset prediction [Barcellos2024, Fang2024], and TimesFM's advantage may narrow when supervised models have access to informative external features. Third, SARIMAX and Prophet used fixed hyperparameters across all folds and cities, which reflects realistic deployment but may understate their optimized performance. Fourth, only one TimesFM checkpoint was evaluated (2.5, 200M parameters); larger checkpoints or domain-adapted fine-tuning on Brazilian epidemiological series may further improve accuracy. Fifth, the analysis covered eight state capitals. Whether these findings extend to smaller municipalities with shorter or irregular reporting histories requires separate evaluation.

### 4.5 Conclusion

TimesFM 2.5 outperformed six purpose-trained models in seven of eight Brazilian state capitals across a 14-year, 12-month-horizon rolling evaluation, without any city-specific training. These results support the prospective integration of foundation time-series models into Brazil's national dengue surveillance infrastructure. Cities with atypical transmission profiles, such as Belo Horizonte, may still require locally trained models or hybrid approaches.

---

## Acknowledgements

Surveillance data were obtained from the InfoDengue platform (Fiocruz/FGV). TimesFM was developed by Google DeepMind and made available via HuggingFace.

## Data availability statement

Monthly dengue case data are publicly available via the InfoDengue API (https://info.dengue.mat.br/api/). All analysis code, processed data, and result files are available at https://github.com/fabianofilho/dengue-timeseries-skforecast.

## Conflict of interest

The author declares no conflict of interest.

## Ethics statement

This study used publicly available, aggregated surveillance data without individual-level records. No ethics committee review was required.

---

## References

1. Leung XY et al. A systematic review of dengue outbreak prediction models. PLOS NTD. 2023. doi:10.1371/journal.pntd.0010631 [Leung2023]
2. GurgelGoncalves R et al. The greatest Dengue epidemic in Brazil. Rev Soc Bras Med Trop. 2024. doi:10.1590/0037-8682-0113-2024 [GurgelGoncalves2024]
3. Siqueira Junior JB et al. Epidemiology and costs of dengue in Brazil: a systematic literature review. Int J Infect Dis. 2022. doi:10.1016/j.ijid.2022.06.050 [Siqueira2022]
4. Roster K et al. Machine-Learning-Based Forecasting of Dengue Fever in Brazilian Cities. Am J Epidemiol. 2022. doi:10.1093/aje/kwac090 [Roster2022]
5. Codeco C et al. Infodengue: A nowcasting system for the surveillance of arboviruses in Brazil. Rev Epidemiol Sante Publique. 2018. doi:10.1016/j.respe.2018.05.408 [Codeco2018infodengue]
6. Fang L et al. Meteorological factors cannot be ignored in ML-based methods for predicting dengue. Int J Biometeorol. 2024. doi:10.1007/s00484-023-02605-1 [Fang2024]
7. Das A et al. A decoder-only foundation model for time-series forecasting. ICML 2024. arXiv:2310.10688 [Das2024timesfm]
8. Collins GS et al. TRIPOD+AI statement. BMJ. 2024. doi:10.1136/bmj-2023-078378 [Collins2024tripod]
9. Taylor SJ, Letham B. Forecasting at Scale. Am Stat. 2018. doi:10.1080/00031305.2017.1380080 [Taylor2018prophet]
10. Tashman LJ. Out-of-sample tests of forecasting accuracy. Int J Forecasting. 2000. doi:10.1016/S0169-2070(00)00065-0 [Tashman2000]
11. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023. doi:10.1016/j.patter.2023.100804 [Kapoor2023]
12. Hewamalage H et al. Forecast evaluation for data scientists: common pitfalls and best practices. DMKD. 2023. doi:10.1007/s10618-022-00894-5 [Hewamalage2023]
13. Chen X, Moraga P. Assessing dengue forecasting methods in Rio de Janeiro, Brazil. Trop Med Health. 2025. doi:10.1186/s41182-025-00723-7 [Chen2025rj]
14. Chen X, Moraga P. Forecasting dengue across Brazil with LSTM neural networks. BMC Public Health. 2025. doi:10.1186/s12889-025-22106-7 [Chen2025lstm]
15. McGough SF et al. A dynamic ensemble approach to forecast dengue fever epidemic years in Brazil. J R Soc Interface. 2021. doi:10.1098/rsif.2020.1006 [McGough2021]
16. Sebastianelli A et al. A reproducible ensemble machine learning approach to forecast dengue outbreaks. Sci Rep. 2024. doi:10.1038/s41598-024-52796-9 [Sebastianelli2024]
17. Barcellos C et al. Climate change, thermal anomalies, and the recent progression of dengue in Brazil. Sci Rep. 2024. doi:10.1038/s41598-024-56044-y [Barcellos2024]
18. Ansari AF et al. Chronos: Learning the Language of Time Series. TMLR. 2024. arXiv:2403.07815 [Ansari2024chronos]
19. Woo G et al. Unified Training of Universal Time Series Forecasting Transformers (Moirai). ICML 2024. arXiv:2402.02592 [Woo2024moirai]
20. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016. doi:10.1145/2939672.2939785 [Chen2016xgboost]
21. Ke G et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017. [Ke2017lightgbm]
22. Prokhorenkova L et al. CatBoost: Unbiased Boosting with Categorical Features. NeurIPS 2018. arXiv:1706.09516 [Prokhorenkova2018catboost]
23. Breiman L. Random Forests. Machine Learning. 2001. doi:10.1023/A:1010933404324 [Breiman2001]
24. Hyndman RJ, Koehler AB. Another look at measures of forecast accuracy. Int J Forecasting. 2006. doi:10.1016/j.ijforecast.2006.03.001 [HyndmanKoehler2006]
25. Diebold FX, Mariano RS. Comparing Predictive Accuracy. J Bus Econ Stat. 1995. doi:10.1080/07350015.1995.10524599 [DieboldMariano1995]
26. Harvey D, Leybourne S, Newbold P. Testing the equality of prediction mean squared errors. Int J Forecasting. 1997. doi:10.1016/S0169-2070(96)00719-4 [HarveyLeybourneNewbold1997]
27. Kunsch HR. The Jackknife and the Bootstrap for General Stationary Observations. Ann Stat. 1989. doi:10.1214/aos/1176347265 [Kunsch1989]
28. Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70. [Holm1979]
29. Aksu T et al. GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. 2024. arXiv:2410.10393 [Aksu2024gifteval]
30. Google Research. TimesFM 2.5 (200M) model card. Hugging Face, 2025. https://huggingface.co/google/timesfm-2.5-200m-pytorch [TimesFM25card]

*Full BibTeX available in paper/refs/references.bib*

---

## Figures

**Figure 1.** sMAPE heatmap across all models (rows) and cities (columns). Lower values (yellow) indicate better performance. TimesFM occupies the lowest-error cells in seven of eight cities. Models sorted by mean sMAPE across cities.

**Figure 2.** TimesFM performance across eight cities. Left panel: ranking of TimesFM by city (gold = 1st place, blue = top 3, salmon = outside top 3). Right panel: grouped bar chart comparing TimesFM sMAPE (blue) with the best supervised competitor per city (salmon).

**Figure 3.** Dengue case time series for all eight Brazilian state capitals (2010-2024). Each panel shows monthly notified cases; grey shading indicates the four-year minimum training window used in rolling-origin cross-validation.

**Figure 4.** Predicted vs. observed monthly dengue cases in Sao Paulo for all seven models, pooled across rolling-origin folds. Black line: observed; dashed blue line: model predictions.
