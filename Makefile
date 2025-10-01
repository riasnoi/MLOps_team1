.PHONY: setup venv dvc-init data preprocess clean lint test

setup: venv dvc-init

venv:
	python3 -m venv .venv || true
	@echo "Run: source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

dvc-init:
	dvc init
	mkdir -p dvc_remote
	dvc remote add -d local_remote dvc_remote

data:
	python src/download_data.py

preprocess:
	dvc repro

clean:
	rm -rf data/processed

lint:
	black --check .
	isort --check-only .
	flake8

test:
	pytest -q
