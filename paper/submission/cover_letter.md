# Cover Letter

Dear Editor,

We submit the manuscript "A Zero-Shot Foundation Model Versus City-Trained Models for 12-Month Dengue Forecasting in Eight Brazilian State Capitals: A Rolling-Origin Benchmark" for consideration as a Research Article in PLOS Neglected Tropical Diseases.

Time-series foundation models can forecast any series without being trained on it. For dengue surveillance in Brazil, where maintaining one model per municipality is costly, this is an attractive property, but whether these models forecast dengue as well as conventional approaches is unclear. We tested that claim for TimesFM 2.5 in eight Brazilian state capitals, using 15 years of InfoDengue data, 121 rolling forecast origins per city, and a 12-month horizon, against a seasonal naive benchmark and six models fitted on each city's history.

With all trained models fitted on log-transformed counts, TimesFM was about as accurate as the best of them (mean sMAPE 75.3% vs 71.8% for CatBoost and 73.3% for Random Forest), had the lowest scaled error in six of eight cities, and was best one month ahead, but only 1 of 56 comparisons was significant after multiplicity correction, and the model failed in Belo Horizonte. At 12 months, no model outperformed the seasonal naive forecast. We also show that fitting the trained models on raw counts, omitting the naive benchmark, and bootstrapping overlapping forecasts as if they were independent would each have made TimesFM look clearly superior.

We believe the manuscript will interest PLOS NTD readers for two reasons: it gives a realistic estimate of what a zero-shot model can offer arboviral surveillance, and it provides a reproducible evaluation protocol that other groups can apply before adopting such models. All code, data, forecasts, and the script that produces every number in the manuscript are publicly available.

The manuscript has not been submitted elsewhere. The author declares no conflict of interest.

Sincerely,

Fabiano Bozza Filho  
Laboratorio de Big Data e Analise Preditiva em Saude (LABDAPS)  
Faculdade de Saude Publica, Universidade de Sao Paulo  
fabiano.nb@gmail.com
