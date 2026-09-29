# Peer-Review Interno — Checklist TRIPOD+AI (v2)

Paper: A Zero-Shot Foundation Model Versus City-Trained Models for 12-Month Dengue Forecasting
Revisão: 2026-09-29 (benchmark v2, branch feat/benchmark-v2)

## O que mudou da v1 (03/05/2026) e por quê

A v1 afirmava superioridade do TimesFM (1º em 7/8 cidades). A revisão encontrou:
- números do texto que não batiam com os CSVs (médias do CatBoost 80,1/82,7 vs 85,6 real; MAE "6/8" vs 8/8; negritos errados na Tabela 3);
- Prophet "89,7" em SP sem nenhum resultado que o sustentasse;
- ICs bootstrap sem script e calculados como se as 1.452 previsões sobrepostas fossem independentes (IC ~4,5x estreito demais);
- Methods em desacordo com o código (interpolação inexistente, versões erradas, TimesFM 2.5 "de 2024");
- ausência de baseline seasonal naive e modelos treinados na escala bruta (árvores não extrapolam acima do máximo do treino).

Com baselines justos (log1p) e inferência por block bootstrap, o TimesFM é comparável, não superior. O paper foi reenquadrado.

## Rigor metodológico

- [x] TRIPOD+AI 2024 declarado
- [x] Seasonal naive como benchmark; MASE além de sMAPE
- [x] Todos os modelos treinados em log1p (decisão registrada antes de ver os resultados v2; escala bruta como sensibilidade)
- [x] IC 95% por moving block bootstrap sobre origens (bloco 12, 2.000 réplicas, seed fixa)
- [x] Diebold-Mariano HAC + correção HLN, Holm sobre 56 comparações
- [x] Sensibilidades: escala bruta (Tabela S3) e sem 2024 (Tabela S4)
- [x] Reprodutibilidade: modelos em escala bruta repetem a v1 (diferença máxima 0,46 pp)
- [x] Todos os números do texto saem de `scripts/analyze_results.py` (`results/v2/summary.json`, `paper/tables/`)
- [ ] Sem margem de equivalência pré-especificada: o paper diz explicitamente que não prova equivalência

## Transparência

- [x] Código, dados processados, previsões e tabelas no repositório
- [x] Versões exatas dos pacotes na seção 2.8
- [ ] Criar a tag `paper-v2` no commit de merge (citada no manuscrito)
- [x] Conflito de interesse e ética declarados

## Anti-leakage

- [x] Rolling origin com janela expansiva; mesmos dados de treino e alvos para todos os modelos
- [x] Nenhuma informação da janela de teste no ajuste ou na configuração
- [x] Contaminação do pré-treino do TimesFM discutida (GIFT-Eval tem CDC FluView e Project Tycho, sem InfoDengue; Wikipedia/Trends sobrepõem o período)
- [x] Dados consolidados (não vintages em tempo real) declarados como limitação

## Versão LaTeX PLOS NTD (2026-09-29)

- [x] Template oficial PLOS (v3.8, abr/2026) com `plos2025.bst`; `.tex` único sem `\input`, bibliografia embutida
- [x] Todo número do texto via `\V{}` a partir de `paper/plos/verified_numbers.json` (109 valores); chave ausente interrompe o build
- [x] Abstract estruturado NTD (275 palavras), author summary (182), título com 149 caracteres
- [x] 28 referências conferidas no Crossref (DOI) e no arXiv; `.bib` gerado a partir dos metadados oficiais
- [x] Figuras em TIFF RGB/LZW 300 dpi dentro dos limites da PLOS (Arial, <= 2250 x 2625 px), fora do PDF
- [x] Declaração de uso de IA nos Acknowledgments
- [x] Compilado (tectonic) e lido: 12 páginas + S1 Appendix (6), 0 overfull, 0 referências indefinidas

## Pendências antes da submissão

1. [feito] Nome e afiliação confirmados: Fabiano Novaes Barcellos Filho, LABDAPS/FSP/USP (a v1 tinha "Bozza" por erro)
2. Coautores? A cover letter e o manuscrito estão com autor único
3. [feito] Tag `paper-v2`
4. Opcional: avaliar os quantis do TimesFM (WIS) e um modelo com covariáveis climáticas (Mosqlimate)
5. [feito] Template PLOS NTD em LaTeX, com nomes próprios acentuados
6. [feito] Funding Statement: sem financiamento
7. [feito] Número de 2024 conferido no texto de Gurgel-Gonçalves 2024: ~6 milhões de casos prováveis e 4.000 óbitos até 15/06/2024 (dado parcial, não anual); frase corrigida
