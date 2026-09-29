PYTHON ?= /opt/miniconda3/envs/dsc/bin/python
UV ?= $(HOME)/.local/bin/uv
JOBS ?= 8

.PHONY: setup fetch-data process-data benchmark-all benchmark-sp analysis figures

setup:
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(UV) pip install --python $(PYTHON) "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"

fetch-data:
	$(PYTHON) scripts/fetch_infodengue.py --output-dir data/raw

process-data: fetch-data
	$(PYTHON) scripts/process_data.py --input-dir data/raw --output-dir data/processed

benchmark-sp:
	PYTHONPATH=src $(PYTHON) scripts/run_benchmark.py \
		--input-csv data/processed/dengue_monthly_sao.csv \
		--output-prefix results/v2/benchmark_sao \
		--horizon 12 \
		--min-train-size 48

# Benchmark v2: 14 modelos, uma cidade por processo (1 thread cada, ver N_JOBS em models.py).
# Retomável: modelos já concluídos ficam em results/v2/cache/.
benchmark-all:
	@mkdir -p results/v2/logs
	@ls data/processed/dengue_monthly_*.csv | sed 's/.*dengue_monthly_//; s/\.csv//' | \
		xargs -n 1 -P $(JOBS) sh -c 'PYTHONPATH=src $(PYTHON) -u scripts/run_benchmark.py \
			--input-csv data/processed/dengue_monthly_$$1.csv --output-prefix results/v2/benchmark_$$1 \
			--horizon 12 --min-train-size 48 > results/v2/logs/$$1.log 2>&1 && echo "[OK] $$1" || echo "[FALHA] $$1"' _

analysis:
	$(PYTHON) scripts/analyze_results.py --results-dir results/v2 --tables-dir paper/tables

figures:
	$(PYTHON) scripts/generate_paper_figures.py --results-dir results/v2 --out-dir paper/figures
