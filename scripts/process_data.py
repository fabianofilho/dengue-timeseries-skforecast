"""
Processa os dados brutos do InfoDengue, agregando os casos semanais em séries temporais mensais para cada cidade.

Input: data/raw/dengue_*.csv
Output: data/processed/dengue_monthly_*.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def process_raw_file(input_path: Path, output_dir: Path) -> None:
    """Carrega um arquivo CSV bruto, agrega para frequência mensal e salva."""
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"[WARN] Falha ao ler {input_path}: {exc}")
        return

    df["date"] = pd.to_datetime(df["date"])
    # Usa 'casos' (notificações observadas), não 'casos_est' (estimativa com nowcasting)
    series = df.set_index("date")["casos"].resample("MS").sum()
    series = series.fillna(0).astype(int)

    # Renomeia para o padrão "data, valor"
    series.index.name = "date"
    series.name = "value"

    city_name = input_path.stem.split("_")[1]
    output_path = output_dir / f"dengue_monthly_{city_name}.csv"
    series.to_csv(output_path)
    print(f"[INFO] Série mensal para {city_name} salva em {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Processa dados brutos de dengue")
    parser.add_argument("--input-dir", default="data/raw", help="Diretório com arquivos brutos")
    parser.add_argument("--output-dir", default="data/processed", help="Diretório de saída")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(input_dir.glob("dengue_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Nenhum arquivo 'dengue_*.csv' encontrado em {input_dir}")

    for file_path in raw_files:
        process_raw_file(file_path, output_dir)


if __name__ == "__main__":
    main()
