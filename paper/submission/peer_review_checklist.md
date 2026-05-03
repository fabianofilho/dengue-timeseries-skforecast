# Peer-Review Interno — Checklist TRIPOD+AI

Paper: Zero-Shot Foundation Models for Dengue Forecasting: A Multi-City Benchmark
Data: 2026-05-03

---

## Rigor metodologico

- [x] Reporting guideline aplicada: TRIPOD+AI 2024 (Collins et al., BMJ 2024, doi:10.1136/bmj-2023-078378) — declarado na abertura de Methods
- [x] Sample size: 180 observacoes por cidade, 121 folds, 1.452 previsoes por modelo por cidade — justificado na secao 2.4
- [x] Missing data: interpolacao linear para semanas faltantes (<0.5%) declarado na secao 2.1
- [x] Multiple comparisons: 7 modelos comparados; nao foi feito teste de significancia estatistica (benchmarks descritivos) — limitacao aceitavel para este tipo de estudo, pode ser mencionada
- [x] IC 95%: AUSENTE nas metricas. **ACAO NECESSARIA:** calcular IC 95% por bootstrap para MAE, RMSE, sMAPE em pelo menos uma cidade representativa
- [x] Validacao externa: nao aplicavel (nao e estudo de desenvolvimento de modelo clinico); o rolling-origin CV ja e validacao temporal prospectiva
- [x] Train/test split: rolling-origin sem leakage temporal — confirmado (Kapoor 2023 citado)

**Pendencia critica: IC 95% para as metricas principais.**

---

## Transparencia

- [x] Codigo disponivel: https://github.com/fabianofilho/dengue-timeseries-skforecast
- [x] Dados: InfoDengue API publica — link na secao Data availability
- [x] Pre-registro: nao aplicavel (estudo retrospectivo descritivo)
- [x] Conflito de interesse: declarado (nenhum)
- [x] Aprovacao etica: declarada (dados publicos agregados, dispensado)
- [ ] **Versao exata do codigo (commit hash):** adicionar o hash do commit atual no manuscrito

---

## Clareza

- [x] Abstract reflete o manuscrito: sim, numeros conferem com Tables 2 e 3
- [x] Numeros consistentes: sMAPE, MAE, RMSE iguais entre abstract, texto e tabelas
- [x] Figuras com caption: sim (4 figuras descritas)
- [x] Acronimos definidos no 1o uso: sMAPE, MAE, RMSE, SINAN, CV, SARIMAX — verificar Prophet (definido?), InfoDengue (definido)
- [x] Discussion nao supera os dados: sim, linguagem cautelosa

**Pendencia menor:** Prophet nao e expandido na primeira ocorrencia. Adicionar "(Aiche Bayesian Decomposition model, Meta)" ou similar.

---

## Anti-leakage (especifico ML/IA)

- [x] Train/test split claro: rolling-origin, min_train=48, horizon=12
- [x] Sem leakage temporal: cada fold usa apenas dados anteriores ao periodo previsto
- [x] Feature engineering: apenas lags autoregressivos da propria serie (sem info do test)
- [x] Hyperparameter tuning: fixo (nao tuned por fold) — declarado explicitamente
- [x] Calibration calculada no test: metricas calculadas nas previsoes out-of-sample
- [x] Kapoor & Narayanan 2023 citado

---

## Etica e fairness

- [ ] Performance por subgrupos demograficos: NAO APLICAVEL (dados agregados por cidade, sem desagregacao por idade/sexo/raca)
- [x] Discussao de potencial vies: under-reporting, fixed hyperparameters — secao 4.4
- [x] Limitacoes de generalizabilidade: municipios menores, populacoes rurais — secao 4.4

---

## Pendencias antes da submissao

### Critica (bloqueia submissao)
1. **IC 95% bootstrap** para sMAPE, MAE, RMSE por cidade e modelo
2. **Commit hash** do repositorio no manuscrito (secao 2.6)
3. **Afiliacao institucional** do autor

### Importante (fortemente recomendada)
4. **Prophet para todas as cidades**: instalar a dependencia e re-rodar o benchmark para eliminar "n/a" da Table 2
5. **Tabela 1 descritivos exatos**: CV calculado com formula correta (verificar se e std/mean ou outro)
6. **Word count final**: preencher o campo no cabecalho

### Opcional (melhora qualidade)
7. Adicionar teste de Diebold-Mariano ou Wilcoxon para diferenca entre TimesFM e CatBoost (especialmente Sao Paulo, onde a margem e de apenas 0.7 pp)
8. Figura de calibracao: residuos ao longo do tempo para TimesFM (verifica drift)
9. Supplementary: tabela completa com LightGBM e Prophet para todas as cidades
