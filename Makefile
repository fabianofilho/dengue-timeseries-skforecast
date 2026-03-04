PYTHON ?= python3
PIP ?= pip3

.PHONY: setup fetch-data process-data benchmark-all benchmark-sp

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

fetch-data:
	$(PYTHON) scripts/fetch_infodengue.py --output-dir data/raw

process-data: fetch-data
	$(PYTHON) scripts/process_data.py --input-dir data/raw --output-dir data/processed

benchmark-sp:
	PYTHONPATH=src $(PYTHON) scripts/run_benchmark.py \
		--input-csv data/processed/dengue_monthly_sao.csv \
		--output-prefix results/benchmark_sao_paulo \
		--horizon 12 \
		--min-train-size 48

benchmark-all:
	@echo "Running benchmark for all cities..."
	@for city_csv in data/processed/dengue_monthly_*.csv; do \
		city_name=$$(basename $$city_csv .csv | sed 's/dengue_monthly_//'); \
		echo "--> Running for $$city_name"; \
		PYTHONPATH=src $(PYTHON) scripts/run_benchmark.py \
			--input-csv $$city_csv \
			--output-prefix results/benchmark_$$city_name \
			--horizon 12 \
			--min-train-size 48; \
	done
