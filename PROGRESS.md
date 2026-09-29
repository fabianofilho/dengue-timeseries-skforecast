# PROGRESS — dengue-timeseries-skforecast

Atualizado em 2026-09-29 (versão LaTeX PLOS).

## Objetivo
Paper para a PLOS NTD comparando o TimesFM 2.5 zero-shot com modelos treinados por cidade na previsão mensal de dengue em 8 capitais, com horizonte de 12 meses.

## Estado atual
- Benchmark v2 mergeado (PR #1, tag `paper-v2`): 14 modelos x 8 cidades, 121 origens.
- Conclusão v2: o TimesFM é comparável, não superior (sMAPE médio 75,3% vs CatBoost-log 71,8%); só 1 de 56 comparações é significativa após Holm; falha em Belo Horizonte; em 12 meses nenhum modelo vence o seasonal naive.
- Manuscrito (`paper/manuscript.md`) reenquadrado; todo número sai de `scripts/analyze_results.py`.

- Manuscrito no template LaTeX da PLOS NTD em `paper/plos/` (`make paper`), com PDF, S1 Appendix e Fig1-4.tif prontos.

## Próximos passos
1. Definir coautores (hoje o autor é único) e preencher o Funding Statement (`paper/plos/submission_fields.md`)
2. Conferir na fonte o número de 6 milhões de casos em 2024 (Gurgel-Gonçalves 2024)
3. Opcional: avaliar os quantis do TimesFM (WIS) e covariáveis climáticas (Mosqlimate)
4. Opcional: testar Chronos e Moirai

## Decisões e por quê
- Modelos treinados em log1p como análise primária (fixado antes de ver a v2): na escala bruta, as árvores não extrapolam acima do máximo do treino.
- IC por moving block bootstrap sobre as origens (bloco 12): as previsões se sobrepõem; o bootstrap iid dá IC 3,4 a 4,9x estreito demais.
- `DENGUE_N_JOBS=1` por padrão: o XGBoost multithread trava no macOS ARM depois do libomp do LightGBM/torch; paralelizar por cidade (`make benchmark-all`).

## Restrições
- Rodar no env `/opt/miniconda3/envs/dsc` (Python 3.10, timesfm 2.0.0, skforecast 0.20.1).
- O macOS não tem `timeout`, e `xargs -I` tem limite de 255 bytes (por isso o Makefile usa `-n 1`).

## Descartado
- Afirmar superioridade do TimesFM (enquadramento da v1).
- O Prophet "89,7%" da v1 (sem resultado que o sustentasse).
