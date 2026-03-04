"""
Coleta dados de dengue via API do InfoDengue (info.dengue.mat.br).

Municípios selecionados (capitais e grandes cidades do Brasil):
  - São Paulo (SP):       3550308
  - Rio de Janeiro (RJ):  3304557
  - Belo Horizonte (MG):  3106200
  - Brasília (DF):        5300108
  - Fortaleza (CE):       2304400
  - Recife (PE):          2611606
  - Manaus (AM):          1302603
  - Salvador (BA):        2927408

Período: 2010 a 2024 (semanas epidemiológicas)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

CITIES = {
    "sao_paulo":      {"geocode": 3550308, "uf": "SP"},
    "rio_de_janeiro": {"geocode": 3304557, "uf": "RJ"},
    "belo_horizonte": {"geocode": 3106200, "uf": "MG"},
    "brasilia":       {"geocode": 5300108, "uf": "DF"},
    "fortaleza":      {"geocode": 2304400, "uf": "CE"},
    "recife":         {"geocode": 2611606, "uf": "PE"},
    "manaus":         {"geocode": 1302603, "uf": "AM"},
    "salvador":       {"geocode": 2927408, "uf": "BA"},
}

BASE_URL = "https://info.dengue.mat.br/api/alertcity"


def fetch_city(geocode: int, ey_start: int = 2010, ey_end: int = 2024) -> pd.DataFrame:
    """Busca todos os registros semanais de dengue para um município."""
    all_records: list[dict] = []
    for year in range(ey_start, ey_end + 1):
        url = (
            f"{BASE_URL}?geocode={geocode}&disease=dengue&format=json"
            f"&ew_start=1&ew_end=53&ey_start={year}&ey_end={year}"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            records = resp.json()
            if isinstance(records, list):
                all_records.extend(records)
        except Exception as exc:
            print(f"  [WARN] Falha ao buscar geocode={geocode} ano={year}: {exc}")
        time.sleep(0.3)  # respeitar rate limit

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # data_iniSE é timestamp em milissegundos
    df["date"] = pd.to_datetime(df["data_iniSE"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta dados InfoDengue")
    parser.add_argument("--output-dir", default="data/raw", help="Diretório de saída")
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--year-end",   type=int, default=2024)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for city_name, info in CITIES.items():
        geocode = info["geocode"]
        print(f"[INFO] Buscando {city_name} (geocode={geocode})...")
        df = fetch_city(geocode, ey_start=args.year_start, ey_end=args.year_end)
        if df.empty:
            print(f"  [WARN] Sem dados para {city_name}")
            continue

        out_path = out_dir / f"dengue_{city_name}_{args.year_start}_{args.year_end}.csv"
        df.to_csv(out_path, index=False)
        print(f"  [OK] {len(df)} semanas salvas em {out_path}")
        summary.append({
            "city": city_name,
            "geocode": geocode,
            "uf": info["uf"],
            "n_weeks": len(df),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "total_casos": int(df["casos"].sum()),
        })

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Resumo salvo em {summary_path}")


if __name__ == "__main__":
    main()
