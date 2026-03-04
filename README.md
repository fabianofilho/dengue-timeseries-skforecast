# Predição de Casos de Dengue no Brasil com Skforecast

Análise de séries temporais de casos de dengue em diversas capitais brasileiras, utilizando dados do **[InfoDengue](https://info.dengue.mat.br/)** e modelos de machine learning com a biblioteca **[skforecast](https://skforecast.org/)**.

## Objetivo

Avaliar a capacidade de diferentes modelos de forecasting em prever o número mensal de casos de dengue, utilizando uma metodologia de backtesting com **rolling origin** para uma avaliação robusta e sem vazamento de dados.

Modelos comparados:

1. **SARIMAX** — baseline estatístico com sazonalidade multiplicativa
2. **LightGBM** — gradient boosting de alta performance
3. **XGBoost** — implementação otimizada de gradient boosting
4. **CatBoost** — gradient boosting com tratamento nativo de features categóricas
5. **RandomForest** — ensemble de árvores de decisão
6. **Ridge** — regressão linear com regularização L2

## Dados

- **Fonte**: [InfoDengue](https://info.dengue.mat.br/) — Sistema de Alerta de Arboviroses (Fiocruz/FGV)
- **Período**: Janeiro/2010 a Dezembro/2024 (dados semanais agregados mensalmente)
- **Cidades**: São Paulo (SP), Rio de Janeiro (RJ), Belo Horizonte (MG), Brasília (DF), Fortaleza (CE), Recife (PE), Manaus (AM), Salvador (BA)
- **Extração**: Via API pública do InfoDengue

### Resumo dos Dados

| Cidade | Total de Casos (2010–2024) | Média Mensal |
|---|---:|---:|
| São Paulo | 1.711.689 | ~9.500 |
| Belo Horizonte | 1.086.158 | ~6.000 |
| Brasília | 724.545 | ~4.000 |
| Rio de Janeiro | 638.433 | ~3.500 |
| Fortaleza | 382.040 | ~2.100 |
| Recife | 185.593 | ~1.000 |
| Manaus | 132.282 | ~730 |
| Salvador | 116.817 | ~650 |

## Metodologia

- **Backtesting**: Rolling origin cross-validation
- **Horizonte de previsão**: 12 meses
- **Janela mínima de treino**: 48 meses
- **Métricas**: MAE, RMSE, sMAPE
- **Lags para modelos de ML**: 24

### Configuração dos Modelos

| Modelo | Configuração Principal |
|---|---|
| SARIMAX | order=(1,1,1), seasonal_order=(1,1,0,12) |
| LightGBM | ForecasterRecursive, 24 lags, random_state=42 |
| XGBoost | ForecasterRecursive, 24 lags, objective=reg:squarederror |
| CatBoost | ForecasterRecursive, 24 lags, verbose=0 |
| RandomForest | ForecasterRecursive, 24 lags, n_estimators=100 |
| Ridge | ForecasterRecursive, 24 lags |

## Resultados (São Paulo, 2010–2024)

| Modelo | MAE | RMSE | sMAPE (%) | n_previsões |
|---|---:|---:|---:|---:|
| **CatBoost** | **9.892** | **43.838** | **78.7** | 1.452 |
| XGBoost | 10.686 | 44.490 | 84.8 | 1.452 |
| SARIMAX | 17.031 | 104.020 | 87.1 | 1.452 |
| RandomForest | 10.727 | 43.822 | 87.3 | 1.452 |
| Ridge | 3.862.818 | 99.332.702 | 118.5 | 1.452 |
| LightGBM | 11.225 | 43.380 | 119.8 | 1.452 |

**Observações:**
- **CatBoost** apresentou o melhor desempenho geral (menor MAE e sMAPE).
- Modelos de gradient boosting (CatBoost, XGBoost) superaram o baseline SARIMAX.
- A alta variabilidade da dengue — com picos epidêmicos abruptos — representa um desafio inerente para todos os modelos.
- Ridge e LightGBM apresentaram instabilidade neste dataset, sugerindo necessidade de ajuste de hiperparâmetros.

## Como Usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Baixar e processar os dados

```bash
python scripts/fetch_infodengue.py --output-dir data/raw
python scripts/process_data.py --input-dir data/raw --output-dir data/processed
```

### 3. Executar o benchmark

```bash
# Apenas São Paulo
PYTHONPATH=src python scripts/run_benchmark.py \
    --input-csv data/processed/dengue_monthly_sao.csv \
    --output-prefix results/benchmark_sao_paulo \
    --horizon 12 \
    --min-train-size 48

# Todas as cidades
make benchmark-all
```

### 4. Gerar figuras

```bash
python scripts/generate_figures.py
```

## Estrutura do Repositório

```
.
├── Makefile
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                              # Dados brutos da API InfoDengue
│   └── processed/                        # Séries mensais processadas
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb     # Análise exploratória
│   ├── 02_benchmark_models.ipynb         # Benchmark dos modelos
│   └── 03_forecast_future.ipynb          # Previsão futura com o melhor modelo
├── results/
│   ├── comparisons/                      # Comparações entre cidades
│   └── figures/                          # Figuras geradas
├── scripts/
│   ├── fetch_infodengue.py               # Coleta de dados via API
│   ├── process_data.py                   # Processamento e agregação mensal
│   ├── run_benchmark.py                  # Orquestração do backtesting
│   └── generate_figures.py              # Geração de figuras
└── src/
    └── dengue_forecast/
        ├── __init__.py
        ├── data.py                       # Funções de carregamento de dados
        ├── evaluate.py                   # Métricas e backtesting
        └── models.py                     # Wrappers dos modelos
```

## Próximos Passos

1. Adicionar variáveis exógenas (temperatura, umidade, índice pluviométrico) via API Mosqlimate
2. Otimizar hiperparâmetros com `optuna` e `skforecast`
3. Implementar previsão probabilística com intervalos de confiança
4. Expandir o benchmark para todas as capitais
5. Desenvolver modelo global multi-série com `ForecasterRecursiveMultiSeries`

## Referências

- **InfoDengue**: Sistema de Alerta de Arboviroses — Fiocruz/FGV — [info.dengue.mat.br](https://info.dengue.mat.br/)
- **skforecast**: Biblioteca para forecasting com ML — [skforecast.org](https://skforecast.org/)
- Codeço, C.T. et al. (2018). Estimating the effective reproduction number of dengue. *Epidemics*, 25, 101–111.
- Bastos, L.S. et al. (2019). A modelling approach for correcting reporting delays in disease surveillance data. *Statistics in Medicine*, 38(22), 4363–4377.
