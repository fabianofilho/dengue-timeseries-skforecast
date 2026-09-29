# Predição de Casos de Dengue no Brasil com Skforecast

Análise de séries temporais de casos de dengue em diversas capitais brasileiras, utilizando dados do **[InfoDengue](https://info.dengue.mat.br/)** e modelos de machine learning com a biblioteca **[skforecast](https://skforecast.org/)**.

## Objetivo

Avaliar a capacidade de diferentes modelos de forecasting em prever o número mensal de casos de dengue, utilizando uma metodologia de backtesting com **rolling origin** para uma avaliação robusta e sem vazamento de dados.

Modelos comparados (benchmark v2):

1. **Seasonal naive** — repete o mesmo mês do último ano (benchmark de referência)
2. **SARIMA** (1,1,1)(1,1,0)12 — estatístico
3. **Prophet** (Meta)
4. **LightGBM**, 5. **XGBoost**, 6. **CatBoost**, 7. **Random Forest** — `ForecasterRecursive` do skforecast, 24 lags
8. **TimesFM 2.5** (Google Research) — modelo de fundação, zero-shot

Todos os modelos treinados são ajustados em `log1p(y)` (análise primária) e também na escala bruta (sensibilidade).

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

- **Backtesting**: rolling origin com janela expansiva, 121 origens por cidade (jan/2014 a jan/2024)
- **Horizonte**: 12 meses; treino mínimo de 48 meses; 1.452 previsões por modelo e cidade
- **Métricas**: sMAPE (primária), MASE, MAE, RMSE
- **Inferência**: moving block bootstrap sobre origens (bloco 12, 2.000 réplicas) e Diebold-Mariano (HAC + HLN) com Holm
- **Sensibilidades**: escala bruta e janelas que terminam até dez/2023 (sem a epidemia de 2024)

## Resultados (v2, 8 capitais, 2014-2024)

sMAPE médio entre as 8 cidades (análise primária, modelos treinados em log1p):

| Modelo | sMAPE médio (%) | MASE médio | 1º lugar sMAPE | 1º lugar MASE |
|---|---:|---:|---:|---:|
| CatBoost | **71,8** | 1,37 | 3 | 1 |
| Random Forest | 73,3 | 1,39 | 1 | 0 |
| TimesFM (zero-shot) | 75,3 | **1,33** | 2 | 6 |
| XGBoost | 78,7 | 1,47 | 0 | 0 |
| Seasonal naive | 79,3 | 1,53 | 1 | 0 |
| Prophet | 80,1 | 1,45 | 1 | 1 |
| LightGBM | 81,6 | 1,47 | 0 | 0 |
| SARIMA | 83,5 | 2,16 | 0 | 0 |

- O TimesFM, sem treino nenhum, fica **comparável** aos melhores modelos treinados, mas **não é superior**: só 1 das 56 comparações é significativa após Holm (TimesFM melhor que SARIMA em Recife).
- O TimesFM é o pior modelo em Belo Horizonte (97,1% vs 62,0% do CatBoost).
- Em 1 mês o TimesFM é o melhor; em 12 meses nenhum modelo vence o seasonal naive.
- Na escala bruta (configuração da v1), as árvores perdem para o seasonal naive em 6 de 8 cidades, e é isso que fazia o TimesFM parecer superior.

Tabelas completas em `paper/tables/`, números em `results/v2/summary.json`, figuras em `paper/figures/`. Os resultados da v1 (maio/2026) continuam em `results/benchmark_*` para comparação.

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

### 3. Executar o benchmark, a análise e as figuras

```bash
make benchmark-all   # 8 cidades em paralelo, ~40 min num Apple M4; retomável via results/v2/cache
make analysis        # métricas, ICs, testes e tabelas (paper/tables, results/v2)
make figures         # figuras do manuscrito (paper/figures)
make paper           # manuscrito LaTeX PLOS NTD: tabelas, números verificados, .tex único, PDFs e TIFFs (paper/plos)
```

O texto do manuscrito é editado em `paper/plos/manuscript_src.tex`, sem números digitados: cada valor entra como `\V{chave}` e sai de `paper/plos/verified_numbers.json`. O arquivo de submissão (`paper/plos/manuscript.tex`) é gerado. O que vai em cada campo do Editorial Manager está em `paper/plos/submission_fields.md`.

Os modelos rodam com 1 thread cada (`DENGUE_N_JOBS`, padrão 1): no macOS ARM o XGBoost multithread trava depois que LightGBM e torch carregam outro `libomp`.

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
├── paper/
│   ├── manuscript.md                     # Manuscrito (PLOS NTD)
│   ├── figures/ tables/                  # Gerados por make figures / make analysis
│   ├── refs/references.bib
│   └── submission/                       # Cover letter, highlights, checklist
├── results/
│   ├── benchmark_*                       # v1 (maio/2026), mantidos para reprodutibilidade
│   ├── figures/                          # Figuras exploratórias da v1
│   └── v2/                               # Previsões, métricas, testes e summary.json da v2
├── scripts/
│   ├── fetch_infodengue.py               # Coleta de dados via API
│   ├── process_data.py                   # Processamento e agregação mensal
│   ├── run_benchmark.py                  # Backtesting rolling origin (14 modelos)
│   ├── analyze_results.py                # Métricas, ICs, testes e tabelas do paper
│   ├── generate_paper_figures.py         # Figuras do manuscrito
│   └── generate_figures.py               # Figuras exploratórias da v1
└── src/
    └── dengue_forecast/
        ├── __init__.py
        ├── data.py                       # Funções de carregamento de dados
        ├── evaluate.py                   # Métricas e backtesting
        └── models.py                     # Wrappers dos modelos
```

## Próximos Passos

1. Avaliar as previsões probabilísticas (quantis do TimesFM, WIS)
2. Adicionar covariáveis climáticas via API Mosqlimate
3. Otimizar hiperparâmetros dos modelos treinados
4. Testar outros modelos de fundação (Chronos, Moirai) e fine-tuning
5. Expandir para municípios menores

## Referências

- **InfoDengue**: Sistema de Alerta de Arboviroses — Fiocruz/FGV — [info.dengue.mat.br](https://info.dengue.mat.br/)
- **skforecast**: Biblioteca para forecasting com ML — [skforecast.org](https://skforecast.org/)
- Codeço, C.T. et al. (2018). Estimating the effective reproduction number of dengue. *Epidemics*, 25, 101–111.
- Bastos, L.S. et al. (2019). A modelling approach for correcting reporting delays in disease surveillance data. *Statistics in Medicine*, 38(22), 4363–4377.
