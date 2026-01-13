# IoT Time-Series Forecasting (MLOps-ready)



An end-to-end time-series forecasting project that demonstrates industry ML engineering skills:

training pipeline, evaluation, model artifact, FastAPI inference, Docker, and CI.



## Problem

Forecast sensor values from IoT devices to support predictive maintenance and anomaly prevention.



## Tech stack

Python, PyTorch, FastAPI, Docker, GitHub Actions (CI), pytest, ruff



## Dataset

- Default: ETT-small (ETTh1.csv)

- If download fails (offline), the project auto-generates a synthetic dataset.



## Quickstart (local)



### 1) Create environment

```bash

python -m venv .venv

source .venv/bin/activate

pip install -e ".[dev]"


