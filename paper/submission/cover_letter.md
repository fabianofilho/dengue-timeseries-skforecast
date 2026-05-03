# Cover Letter

Dear Editor,

We submit our manuscript "Zero-Shot Foundation Models for Dengue Forecasting: A Multi-City Benchmark Across Brazilian State Capitals" for consideration as an original research article in PLOS Neglected Tropical Diseases.

Dengue fever affects millions of Brazilians each year, with 2024 recording more than 6 million probable cases, the highest burden on record. Accurate long-range forecasting of monthly case counts is critical for planning vaccination campaigns, vector control operations, and hospital capacity. Yet most available forecasting tools require city-specific training and frequent retraining as epidemic dynamics evolve, creating a maintenance burden that limits scalability across the hundreds of municipalities covered by Brazil's surveillance network.

We address this problem by evaluating TimesFM 2.5, a 200-million-parameter zero-shot foundation model developed by Google DeepMind, against six established forecasting models across eight Brazilian state capitals, using 14 years of monthly notification data from the InfoDengue surveillance platform. Our rolling-origin benchmark, designed explicitly to prevent temporal data leakage, shows that TimesFM ranked first in seven of eight cities by symmetric MAPE without any city-specific training. Bootstrap confidence intervals confirm the advantage is statistically meaningful in most settings, though the margin in Sao Paulo (0.7 percentage points over CatBoost) does not reach reliable significance.

These findings have direct implications for Brazil's national surveillance infrastructure: a zero-shot foundation model could serve as a low-maintenance baseline forecasting layer within platforms like InfoDengue, reducing the operational overhead of dengue preparedness planning at national scale.

The manuscript has not been submitted elsewhere. All authors contributed substantially and approved the final version. We declare no conflict of interest. Code and data are publicly available at https://github.com/fabianofilho/dengue-timeseries-skforecast.

Sincerely,

Fabiano Bozza Filho  
[Institution: to complete]  
fabiano.nb@gmail.com
