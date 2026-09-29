"""
Gera os assets do manuscrito LaTeX no template PLOS NTD e monta o arquivo de submissão.

Fluxo (rodar depois de scripts/analyze_results.py):
  results/v2/*.csv + data/processed ──> paper/plos/tables/*.tex        (tabelas geradas)
                                    ├──> paper/plos/verified_numbers.json (todo número citado)
                                    ├──> paper/plos/S1_Appendix.tex/.pdf (tabelas suplementares)
                                    └──> paper/plos/manuscript.tex/.pdf  (arquivo único de submissão)

O texto é escrito em paper/plos/manuscript_src.tex e NUNCA traz número digitado: todo valor
entra como \\V{chave}, substituído aqui pelo valor de verified_numbers.json. Chave inexistente
interrompe o build. O arquivo final é achatado (sem \\input), com a bibliografia embutida,
como a PLOS exige; figuras não entram no PDF (vão como TIFF separados).

Exemplo:
  python scripts/build_paper_assets.py            # gera tudo e compila com tectonic
  python scripts/build_paper_assets.py --no-pdf   # só assets e .tex
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_results import CITY_LABELS, PRIMARY_MODELS, RAW_MODELS, REGIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v2"
PLOS = ROOT / "paper" / "plos"
TAB = PLOS / "tables"
TECTONIC = "/opt/miniconda3/envs/tectonic/bin/tectonic"
HEADER = "% gerado por scripts/build_paper_assets.py -- nao editar a mao\n"

CITY_TEX = {
    "sao": "S\\~ao Paulo",
    "rio": "Rio de Janeiro",
    "belo": "Belo Horizonte",
    "brasilia": "Bras\\'ilia",
    "fortaleza": "Fortaleza",
    "recife": "Recife",
    "manaus": "Manaus",
    "salvador": "Salvador",
}
MODEL_TEX = {
    "timesfm": "TimesFM",
    "catboost_log": "CatBoost",
    "xgboost_log": "XGBoost",
    "randomforest_log": "RF",
    "lgbm_log": "LightGBM",
    "sarimax_log": "SARIMA",
    "prophet_log": "Prophet",
    "seasonal_naive": "Naive",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "randomforest": "RF",
    "lgbm": "LightGBM",
    "sarimax": "SARIMA",
    "prophet": "Prophet",
}
MODEL_WORD = {  # nomes por extenso usados na prosa
    "timesfm": "TimesFM",
    "catboost_log": "CatBoost",
    "xgboost_log": "XGBoost",
    "randomforest_log": "Random Forest",
    "lgbm_log": "LightGBM",
    "sarimax_log": "SARIMA",
    "prophet_log": "Prophet",
    "seasonal_naive": "the seasonal naive forecast",
}
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "last"}
NUMWORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def f1(x: float) -> str:
    return f"{x:.1f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def fint(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def signed(x: float) -> str:
    return f"{x:+.1f}".replace("-", "$-$").replace("+", "+")


def ci(lo: float, hi: float) -> str:
    return f"{lo:.1f} to {hi:.1f}".replace("-", "$-$")


def city_list(keys: list[str]) -> str:
    names = [CITY_TEX[k] for k in keys]
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


# --------------------------------------------------------------------------- dados
def load() -> dict:
    d = {
        "m": pd.read_csv(RES / "metrics_primary.csv"),
        "mr": pd.read_csv(RES / "metrics_raw.csv"),
        "mp": pd.read_csv(RES / "metrics_pre2024.csv"),
        "t": pd.read_csv(RES / "tests_primary.csv"),
        "tr": pd.read_csv(RES / "tests_raw.csv"),
        "tp": pd.read_csv(RES / "tests_pre2024.csv"),
        "h": pd.read_csv(RES / "by_horizon_primary.csv"),
        "repro": pd.read_csv(RES / "reproducibility_v1_v2.csv"),
        "ciw": pd.read_csv(RES / "ci_width_iid_vs_block.csv"),
        "summary": json.loads((RES / "summary.json").read_text()),
    }
    series = {}
    for c in CITY_LABELS:
        series[c] = pd.read_csv(ROOT / f"data/processed/dengue_monthly_{c}.csv", index_col=0, parse_dates=True)["value"]
    d["series"] = series
    raw_weeks = json.loads((ROOT / "data/raw/summary.json").read_text())
    d["raw_weeks"] = raw_weeks
    return d


# --------------------------------------------------------------------------- números citados
def verified_numbers(d: dict) -> dict[str, str]:
    m, mr, mp, t, tr, tp, h = d["m"], d["mr"], d["mp"], d["t"], d["tr"], d["tp"], d["h"]
    V: dict[str, str] = {}

    # desenho
    V["n-origins"] = str(d["summary"]["n_origins_full"])
    V["n-origins-pre2024"] = str(d["summary"]["n_origins_pre2024"])
    V["n-forecasts"] = fint(m["n_predictions"].iloc[0])
    V["n-months"] = str(len(next(iter(d["series"].values()))))
    V["n-tests"] = str(len(t))
    V["n-boot"] = fint(d["summary"]["bootstrap"]["n_boot"])
    V["block"] = str(d["summary"]["bootstrap"]["block"])
    weeks = {w["n_weeks"] for w in d["raw_weeks"]}
    assert len(weeks) == 1, weeks
    V["n-weeks"] = str(weeks.pop())

    # descritivos
    means = {c: s.mean() for c, s in d["series"].items()}
    cvs = {c: 100 * s.std() / s.mean() for c, s in d["series"].items()}
    lo_c, hi_c = min(means, key=means.get), max(means, key=means.get)
    V["mean-min-city"], V["mean-min"] = CITY_TEX[lo_c], fint(means[lo_c])
    V["mean-max-city"], V["mean-max"] = CITY_TEX[hi_c], fint(means[hi_c])
    V["cv-min"], V["cv-max"] = fint(min(cvs.values())), fint(max(cvs.values()))
    allmax = {c: (s.max(), s.idxmax()) for c, s in d["series"].items()}
    cmax = max(allmax, key=lambda c: allmax[c][0])
    V["max-count"], V["max-count-city"], V["max-count-year"] = fint(allmax[cmax][0]), CITY_TEX[cmax], str(allmax[cmax][1].year)
    assert all(s.mean() > s.median() for s in d["series"].values())

    # médias primárias
    mean = m.groupby("model")["smape"].mean()
    for k in PRIMARY_MODELS:
        V[f"smape-mean.{k}"] = f1(mean[k])
    mase = m.groupby("model")["mase"].mean()
    for k in PRIMARY_MODELS:
        V[f"mase-mean.{k}"] = f2(mase[k])
    order = mean.sort_values().index.tolist()
    V["order-primary"] = ", ".join(f"{MODEL_WORD[k]} ({f1(mean[k])}\\%)" for k in order)
    firsts = m[m["rank_smape"] == 1].groupby("model")["city"].apply(list).to_dict()
    firsts_mase = m[m["rank_mase"] == 1].groupby("model")["city"].apply(list).to_dict()
    V["first-smape.timesfm"] = str(len(firsts.get("timesfm", [])))
    V["first-mase.timesfm"] = str(len(firsts_mase.get("timesfm", [])))
    V["first-mase.timesfm-word"] = NUMWORD[len(firsts_mase.get("timesfm", []))]
    V["first-smape.catboost-word"] = NUMWORD[len(firsts.get("catboost_log", []))]
    V["first-smape.catboost-cities"] = city_list([c for c in CITY_LABELS if c in firsts.get("catboost_log", [])])
    V["first-smape.timesfm-word"] = NUMWORD[len(firsts.get("timesfm", []))]
    V["first-smape.timesfm-cities"] = city_list([c for c in CITY_LABELS if c in firsts.get("timesfm", [])])
    singles = [f"{MODEL_WORD[k]} ({CITY_TEX[v[0]]})" for k, v in firsts.items() if len(v) == 1 and k not in ("timesfm",)]
    V["first-smape.others"] = city_list_raw(singles)
    tf = m[m["model"] == "timesfm"].set_index("city")
    by_rank: dict[int, list[str]] = {}
    for c in CITY_LABELS:
        by_rank.setdefault(int(tf.loc[c, "rank_smape"]), []).append(c)
    parts = [f"{ORDINAL[r] if r < len(PRIMARY_MODELS) else 'last'} in {city_list(cs)}" for r, cs in sorted(by_rank.items()) if r != 1]
    V["timesfm-rank-text"] = "; ".join(parts)

    # SP como exemplo de IC
    sp = m[m["city"] == "sao"].set_index("model")
    for k in ["catboost_log", "timesfm"]:
        V[f"sp-smape.{k}"] = f1(sp.loc[k, "smape"])
        V[f"sp-smape-ci.{k}"] = ci(sp.loc[k, "smape_lo"], sp.loc[k, "smape_hi"])

    # MASE > 1 para todos os modelos
    piv_mase = m.pivot(index="city", columns="model", values="mase")
    above = [c for c in CITY_LABELS if (piv_mase.loc[c] > 1).all()]
    V["mase-above1-cities"] = city_list(above)
    mae_best = m.loc[m.groupby("city")["mae"].idxmin()].set_index("city")["model"]
    V["mae-first.timesfm-word"] = NUMWORD[int((mae_best == "timesfm").sum())]
    V["mae-first.others"] = "; ".join(
        [f"{MODEL_WORD[k]} in {city_list([c for c in CITY_LABELS if mae_best[c] == k])}" for k in dict.fromkeys(mae_best) if k != "timesfm"]
    )

    # comparações pareadas
    excl = t[(t["diff_hi"] < 0) | (t["diff_lo"] > 0)]
    V["n-ci-excl0"] = str(len(excl))
    fav_t = excl[excl["diff_smape"] < 0]
    fav_c = excl[excl["diff_smape"] > 0]
    V["n-fav-timesfm"] = str(len(fav_t))
    V["n-fav-comp"] = str(len(fav_c))
    V["n-fav-timesfm-word"] = NUMWORD.get(len(fav_t), str(len(fav_t))).capitalize()
    V["n-fav-comp-word"] = NUMWORD.get(len(fav_c), str(len(fav_c)))
    grp = []
    for c in CITY_LABELS:
        comps = fav_t[fav_t["city"] == c]["comparator"].tolist()
        if comps:
            grp.append(f"{city_list_raw([MODEL_WORD[x].replace('the seasonal naive forecast', 'the seasonal naive forecast') for x in comps])} in {CITY_TEX[c]}")
    V["fav-timesfm-list"] = "; ".join(grp)
    comp_cities = sorted(set(fav_c["city"]))
    V["fav-comp-cities"] = city_list(comp_cities)
    sig = t[t["p_holm"] < 0.05]
    assert len(sig) == 1, sig
    s = sig.iloc[0]
    V["n-sig-holm"] = str(len(sig))
    V["sig-city"], V["sig-comp"] = CITY_TEX[s["city"]], MODEL_WORD[s["comparator"]]
    V["sig-diff"], V["sig-ci"] = signed(s["diff_smape"]), ci(s["diff_lo"], s["diff_hi"])
    cb = t[t["comparator"] == "catboost_log"].set_index("city")
    others = cb.drop(index="belo")
    V["cb-diff-min"], V["cb-diff-min-city"] = signed(others["diff_smape"].min()), CITY_TEX[others["diff_smape"].idxmin()]
    V["cb-diff-max"], V["cb-diff-max-city"] = signed(others["diff_smape"].max()), CITY_TEX[others["diff_smape"].idxmax()]
    assert ((others["diff_lo"] < 0) & (others["diff_hi"] > 0)).all()
    V["cb-n-others-word"] = NUMWORD[len(others)]
    V["bh-diff"], V["bh-ci"] = f1(cb.loc["belo", "diff_smape"]), ci(cb.loc["belo", "diff_lo"], cb.loc["belo", "diff_hi"])
    V["bh-p-holm"] = f"{cb.loc['belo', 'p_holm']:.3f}"
    bh = m[m["city"] == "belo"].set_index("model")
    V["bh-smape.timesfm"], V["bh-smape.catboost_log"] = f1(bh.loc["timesfm", "smape"]), f1(bh.loc["catboost_log", "smape"])
    assert int(bh.loc["timesfm", "rank_smape"]) == len(PRIMARY_MODELS)

    # horizonte
    med = h.groupby(["model", "step"])["smape"].median().unstack(0)
    for step in [1, 6, 12]:
        for k in ["timesfm", "catboost_log", "randomforest_log", "seasonal_naive"]:
            V[f"h{step}.{k}"] = f1(med.loc[step, k])
    assert med.loc[1].idxmin() == "timesfm" and med.loc[12].idxmin() == "seasonal_naive"

    # sensibilidade: escala bruta
    mean_r = mr.groupby("model")["smape"].mean()
    trained_raw = [k for k in RAW_MODELS if k not in ("timesfm", "seasonal_naive")]
    V["raw-mean-min"], V["raw-mean-min-model"] = f1(mean_r[trained_raw].min()), MODEL_TEX[mean_r[trained_raw].idxmin()]
    V["raw-mean-max"], V["raw-mean-max-model"] = f1(mean_r[trained_raw].max()), MODEL_TEX[mean_r[trained_raw].idxmax()]
    piv_r = mr.pivot(index="city", columns="model", values="smape")
    trees = ["catboost", "xgboost", "randomforest"]
    worse = [c for c in CITY_LABELS if (piv_r.loc[c, trees] > piv_r.loc[c, "seasonal_naive"]).all()]
    V["raw-trees-worse-naive"] = NUMWORD[len(worse)]
    fr = mr[mr["rank_smape"] == 1].groupby("model").size()
    V["raw-first.timesfm"] = NUMWORD[int(fr.get("timesfm", 0))]
    V["raw-first.naive"] = NUMWORD[int(fr.get("seasonal_naive", 0))]
    assert int(fr.drop(["timesfm", "seasonal_naive"], errors="ignore").sum()) == 0
    no_naive = piv_r.drop(columns="seasonal_naive")
    V["raw-no-naive-first.timesfm"] = NUMWORD[int((no_naive.idxmin(axis=1) == "timesfm").sum())]
    impr = {k: mean_r[k] - mean[k + "_log"] for k in trained_raw}
    V["impr.catboost"], V["impr.randomforest"] = fint(impr["catboost"]), fint(impr["randomforest"])
    V["impr.stat-min"], V["impr.stat-max"] = fint(min(impr["sarimax"], impr["prophet"])), fint(max(impr["sarimax"], impr["prophet"]))
    rep = d["repro"]
    worst = rep.loc[rep["abs_diff"].idxmax()]
    V["repro-max"], V["repro-max-model"], V["repro-max-city"] = f"{worst['abs_diff']:.2f}", MODEL_TEX[worst["model"]], CITY_TEX[worst["city"]]
    rest = rep.drop(index=worst.name)["abs_diff"].max()
    assert rest < 1e-4, rest
    V["repro-rest"] = "0.0001"

    # sensibilidade: sem 2024
    mean_p = mp.groupby("model")["smape"].mean()
    for k in ["catboost_log", "randomforest_log", "timesfm", "seasonal_naive"]:
        V[f"pre-mean.{k}"] = f1(mean_p[k])
    top4 = mean.sort_values().index[:4].tolist()
    assert mean_p.sort_values().index[:4].tolist() == top4
    sig_p = tp[(tp["p_holm"] < 0.05) & (tp["diff_smape"] > 0)]
    assert len(sig_p) == 1 and sig_p.iloc[0]["city"] == "belo"
    V["pre-bh-diff"], V["pre-bh-ci"] = f1(sig_p.iloc[0]["diff_smape"]), ci(sig_p.iloc[0]["diff_lo"], sig_p.iloc[0]["diff_hi"])
    mase_p = mp.groupby("model")["mase"].mean()
    assert (mase_p.drop("sarimax_log") < 1).all() and mase_p["sarimax_log"] > 1
    ciw = d["ciw"]["ratio"]
    V["ciw-min"], V["ciw-max"], V["ciw-med"] = f1(ciw.min()), f1(ciw.max()), f1(ciw.median())
    return V


def city_list_raw(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# --------------------------------------------------------------------------- tabelas
def wide(metrics: pd.DataFrame, models: list[str], col: str, fmt, caption: str, label: str, note: str, ci_col: bool = False) -> str:
    size = "\\footnotesize" if ci_col else "\\small"
    size = size + ("\n\\setlength{\\tabcolsep}{3pt}" if ci_col else "")
    lines = [HEADER, "\\begin{table}[!ht]", "\\centering", size,
             f"\\caption{{{caption}}}", "\\begin{tabular}{l" + "r" * len(models) + "}", "\\toprule",
             "City & " + " & ".join(MODEL_TEX[k] for k in models) + " \\\\", "\\midrule"]
    for c in CITY_LABELS:
        sub = metrics[metrics["city"] == c].set_index("model")
        best = sub.loc[models, col].idxmin()
        cells = []
        for k in models:
            v = fmt(sub.loc[k, col])
            if ci_col:
                v = f"{v} ({fmt(sub.loc[k, col + '_lo'])}--{fmt(sub.loc[k, col + '_hi'])})"
            cells.append(f"\\textbf{{{v}}}" if k == best else v)
        lines.append(f"{CITY_TEX[c]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    mean = metrics.groupby("model")[col].mean()
    lines.append("Mean & " + " & ".join(fmt(mean[k]) for k in models) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\begin{{flushleft}} {note}", "\\end{flushleft}",
              f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_descriptives(d: dict) -> str:
    rows = sorted(d["series"].items(), key=lambda kv: -kv[1].sum())
    lines = [HEADER, "\\begin{table}[!ht]", "\\centering", "\\small",
             "\\caption{\\textbf{Monthly notified dengue cases by city, January 2010 to December 2024.}}",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "City & Region & Mean & Median & Max & Min & CV (\\%) & Total \\\\", "\\midrule"]
    for c, s in rows:
        lines.append(f"{CITY_TEX[c]} & {REGIONS[c]} & {fint(s.mean())} & {s.median():,.1f}".replace(",", "{,}")
                     + f" & {fint(s.max())} & {fint(s.min())} & {fint(100 * s.std() / s.mean())} & {fint(s.sum())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\begin{flushleft} Each series has 180 months. CV, coefficient of variation (standard deviation divided by the mean). Source: InfoDengue.",
              "\\end{flushleft}", "\\label{table1}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_paired(t: pd.DataFrame, comps: list[str], label: str, caption: str, note: str) -> str:
    lines = [HEADER, "\\begin{table}[!ht]", "\\centering", "\\small", f"\\caption{{{caption}}}",
             "\\begin{tabular}{l" + "r" * len(comps) + "}", "\\toprule",
             "City & " + " & ".join(f"vs {MODEL_TEX[k]}" for k in comps) + " \\\\", "\\midrule"]
    for c in CITY_LABELS:
        sub = t[t["city"] == c].set_index("comparator")
        cells = []
        for k in comps:
            r = sub.loc[k]
            txt = f"{signed(r['diff_smape'])} ({ci(r['diff_lo'], r['diff_hi'])})"
            cells.append(f"\\textbf{{{txt}}}" if r["p_holm"] < 0.05 else txt)
        lines.append(f"{CITY_TEX[c]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\begin{{flushleft}} {note}", "\\end{flushleft}",
              f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_dm_long(t: pd.DataFrame, label: str) -> str:
    note = ("Difference = TimesFM minus comparator sMAPE, in percentage points (negative favors TimesFM), with 95\\% paired moving block bootstrap interval over forecast origins (blocks of 12, 2,000 replicates). "
            "DM, Diebold-Mariano statistic on origin-level sMAPE losses with Newey-West variance (12 lags) and Harvey-Leybourne-Newbold correction; "
            f"$p$ adjusted with Holm across all {len(t)} comparisons. RF, Random Forest; Naive, seasonal naive.")
    lines = [HEADER, "\\begin{longtable}{llrrr}",
             "\\caption{\\textbf{TimesFM versus each comparator: paired sMAPE difference and Diebold-Mariano test.} " + note + "}\\label{" + label + "}\\\\",
             "\\toprule", "City & Comparator & Difference (95\\% CI) & DM & $p$ (Holm) \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "City & Comparator & Difference (95\\% CI) & DM & $p$ (Holm) \\\\", "\\midrule", "\\endhead"]
    for c in CITY_LABELS:
        for _, r in t[t["city"] == c].iterrows():
            p = "$<$0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
            dm = f"{r['dm_stat']:.2f}".replace("-", "$-$")
            lines.append(f"{CITY_TEX[c]} & {MODEL_TEX[r['comparator']]} & {signed(r['diff_smape'])} ({ci(r['diff_lo'], r['diff_hi'])}) & {dm} & {p} \\\\")
    lines += ["\\bottomrule",
              "\\end{longtable}"]
    return "\n".join(lines) + "\n"


def build_tables(d: dict) -> dict[str, str]:
    m, mr, mp, t = d["m"], d["mr"], d["mp"], d["t"]
    note_models = ("Trained models were fitted on log1p-transformed counts; TimesFM was used zero-shot on raw counts. "
                   "RF, Random Forest; Naive, seasonal naive. Bold, lowest value in the row.")
    tabs = {
        "table1_descriptives": table_descriptives(d),
        "table2_smape": wide(m, PRIMARY_MODELS, "smape", f1,
                             "\\textbf{sMAPE (\\%) by model and city, 12-month rolling-origin evaluation (121 origins, January 2014 to January 2024).}",
                             "table2", note_models + " 95\\% intervals are given in S1 Appendix, Table A."),
        "table3_mase": wide(m, PRIMARY_MODELS, "mase", f2,
                            "\\textbf{MASE by model and city.}", "table3",
                            "MASE, mean absolute error scaled by the in-sample mean absolute error of the seasonal naive forecast in the training window of each origin; values below 1 indicate lower error than that benchmark. " + note_models),
        "table4_paired": table_paired(t, ["catboost_log", "randomforest_log", "seasonal_naive"], "table4",
                                      "\\textbf{Difference in sMAPE between TimesFM and the two leading trained models and the seasonal naive benchmark.}",
                                      "Values are TimesFM minus comparator sMAPE in percentage points (negative favors TimesFM), with 95\\% paired moving block bootstrap intervals over forecast origins. Bold, Diebold-Mariano test significant after Holm correction across all "
                                      + f"{len(t)} comparisons"
                                      + (" (none in this table)" if not (t[t["comparator"].isin(["catboost_log", "randomforest_log", "seasonal_naive"])]["p_holm"] < 0.05).any() else "")
                                      + ". All comparators are shown in S1 Appendix, Table F."),
    }
    si = {
        "tableA_smape_ci": wide(m, PRIMARY_MODELS, "smape", f1,
                                "\\textbf{sMAPE (\\%) with 95\\% moving block bootstrap intervals, primary model set.}", "tabA", note_models, ci_col=True),
        "tableB_mae": wide(m, PRIMARY_MODELS, "mae", lambda x: fint(x), "\\textbf{MAE (cases per month), primary model set.}", "tabB", note_models),
        "tableC_rmse": wide(m, PRIMARY_MODELS, "rmse", lambda x: fint(x), "\\textbf{RMSE (cases per month), primary model set.}", "tabC", note_models),
        "tableD_smape_raw": wide(mr, RAW_MODELS, "smape", f1,
                                 "\\textbf{sMAPE (\\%) with all trained models fitted on untransformed counts (sensitivity analysis).}", "tabD",
                                 "Same design as the primary analysis, without the log1p transformation. Bold, lowest value in the row."),
        "tableE_smape_pre2024": wide(mp, PRIMARY_MODELS, "smape", f1,
                                     "\\textbf{sMAPE (\\%) for the 109 origins whose forecast windows ended by December 2023 (sensitivity analysis).}", "tabE", note_models),
        "tableF_dm": table_dm_long(t, "tabF"),
    }
    return tabs, si


# --------------------------------------------------------------------------- montagem
def substitute_numbers(src: str, V: dict[str, str]) -> str:
    missing = sorted(set(re.findall(r"\\V\{([^}]+)\}", src)) - set(V))
    if missing:
        raise SystemExit(f"Chaves ausentes em verified_numbers.json: {missing}")
    return re.sub(r"\\V\{([^}]+)\}", lambda mm: V[mm.group(1)], src)


def inline_inputs(src: str, base: Path) -> str:
    def repl(mm):
        return (base / (mm.group(1) + ".tex")).read_text()
    return re.sub(r"\\input\{([^}]+)\}", repl, src)


def compile_tex(tex: Path, keep: bool = False) -> None:
    cmd = [TECTONIC, "-X", "compile", tex.name, "--outdir", str(tex.parent / "build")]
    if keep:
        cmd.append("--keep-intermediates")
    (tex.parent / "build").mkdir(exist_ok=True)
    r = subprocess.run(cmd, cwd=tex.parent, capture_output=True, text=True)
    log = r.stdout + r.stderr
    (tex.parent / "build" / f"{tex.stem}.compile.log").write_text(log)
    if r.returncode != 0:
        print(log[-3000:])
        raise SystemExit(f"tectonic falhou em {tex.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)

    d = load()
    V = verified_numbers(d)
    (PLOS / "verified_numbers.json").write_text(json.dumps(V, indent=1, ensure_ascii=False, sort_keys=True))
    tabs, si = build_tables(d)
    for name, body in {**tabs, **si}.items():
        (TAB / f"{name}.tex").write_text(body)

    # S1 Appendix (tabelas suplementares), documento próprio
    appendix = (PLOS / "S1_Appendix_src.tex").read_text()
    appendix = inline_inputs(appendix, PLOS)
    (PLOS / "S1_Appendix.tex").write_text(appendix)

    # Manuscrito: números, tabelas inline, bibliografia embutida
    src = (PLOS / "manuscript_src.tex").read_text()
    flat = inline_inputs(substitute_numbers(src, V), PLOS)
    stage = PLOS / "manuscript.tex"
    stage.write_text(flat)
    if args.no_pdf:
        return
    compile_tex(stage, keep=True)
    bbl = (PLOS / "build" / "manuscript.bbl").read_text()
    final = flat.replace("\\bibliography{references}", bbl)
    stage.write_text(final)
    compile_tex(stage)
    compile_tex(PLOS / "S1_Appendix.tex")
    print(f"[OK] {len(V)} números verificados; PDFs em {PLOS / 'build'}")


if __name__ == "__main__":
    main()
