# IoT Time-Series Forecasting MLOps

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20Forecasting-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black?logo=githubactions&logoColor=white)](https://github.com/features/actions)

End-to-end **IoT time-series forecasting MLOps project** using **PyTorch, FastAPI, Docker, pytest, ruff, and GitHub Actions**.

The project demonstrates a production-style machine learning workflow for forecasting future sensor values from multivariate IoT streams. It includes dataset loading, sliding-window generation, model training, evaluation, model artifact saving, REST API inference, containerization, and CI checks.

---

## Project Goal

Forecast future IoT sensor values to support:

- predictive maintenance,
- anomaly prevention,
- sensor trend monitoring,
- energy and industrial time-series analytics,
- production-ready ML engineering workflows.

---

## Key Features

- PyTorch LSTM forecasting model
- Multivariate time-series windowing
- Config-driven training with YAML
- ETT-small dataset support
- Synthetic dataset fallback when offline
- Train / validation / test split by time order
- Standardization of input and target values
- RMSE and MAE evaluation
- Saved model artifact for inference
- FastAPI REST API with `/health` and `/predict`
- Docker deployment
- Unit tests with pytest
- Code linting with ruff
- GitHub Actions CI pipeline

---

## Repository Structure

```text
iot-timeseries-forecasting-mlops/
├── configs/
│   └── default.yaml
├── docker/
│   └── Dockerfile
├── src/
│   └── ts_forecast/
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── predict.py
│       ├── train.py
│       └── utils.py
├── tests/
│   └── test_data.py
├── pyproject.toml
└── README.md
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| API | FastAPI, Uvicorn |
| Data Processing | NumPy, pandas |
| Config | YAML |
| Testing | pytest |
| Linting | ruff |
| Deployment | Docker |
| CI/CD | GitHub Actions |

---

## Dataset

Default dataset:

```text
ETT-small / ETTh1.csv
```

If the ETT-small download fails, the project automatically generates a synthetic IoT-like multivariate dataset with daily and weekly patterns.

Default configuration:

```yaml
dataset_name: "ett_small"
csv_path: "data/ett_small.csv"
target_col: "OT"
seq_len: 96
horizon: 24
train_ratio: 0.7
val_ratio: 0.15
standardize: true
```

---

## Model

The forecasting model is an LSTM-based neural network:

```text
Input window:  [batch, seq_len, num_features]
        ↓
LSTM encoder
        ↓
Last hidden state
        ↓
Fully connected head
        ↓
Output: [batch, horizon]
```

Default setup:

- lookback window: `96` time steps
- forecast horizon: `24` time steps
- hidden size: `64`
- LSTM layers: `2`
- dropout: `0.1`
- optimizer: Adam
- loss: MSE

---

## Installation

```bash
git clone https://github.com/Xojiakbar/iot-timeseries-forecasting-mlops.git
cd iot-timeseries-forecasting-mlops
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Training

Run training with the default configuration:

```bash
python -m ts_forecast.train --config configs/default.yaml
```

The best model is saved to:

```text
models/model.pt
```

Training history is saved to:

```text
reports/train_history.json
```

---

## Evaluation

Evaluate the trained model:

```bash
python -m ts_forecast.evaluate --config configs/default.yaml
```

Evaluation metrics are saved to:

```text
reports/metrics.json
```

Example metrics:

```json
{
  "rmse": 1.7908693552017212,
  "mae": 1.3147332668304443,
  "n_test_windows": 2596,
  "horizon": 24,
  "target": "OT",
  "dataset": "ett_small"
}
```

---

## Run the API Locally

After training the model, start the FastAPI server:

```bash
uvicorn ts_forecast.api:app --host 0.0.0.0 --port 8000
```

Check service health:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Prediction API

Endpoint:

```http
POST /predict
```

Request body:

```json
{
  "series": [
    [0.12, 0.31, 0.44, 0.28],
    [0.15, 0.35, 0.47, 0.30]
  ]
}
```

The `series` input must be a 2D list shaped:

```text
[seq_len, num_features]
```

Example response:

```json
{
  "horizon": 24,
  "prediction": [0.42, 0.43, 0.45]
}
```

---

## Docker

Build the Docker image:

```bash
docker build -f docker/Dockerfile -t iot-ts-forecast .
```

Run the container:

```bash
docker run --rm -p 8000:8000 iot-ts-forecast
```

The container uses:

```text
MODEL_PATH=/app/models/model.pt
DEVICE=cpu
```

---

## Testing and Linting

Run tests:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

---

## CI/CD

The GitHub Actions workflow runs automatically on push and pull requests. It performs:

1. Python setup
2. Package installation
3. Ruff linting
4. Pytest test execution

---

## Configuration

Main configuration file:

```text
configs/default.yaml
```

You can modify:

- dataset path,
- target column,
- feature columns,
- sequence length,
- forecast horizon,
- training epochs,
- batch size,
- learning rate,
- model hidden size,
- number of LSTM layers,
- device selection,
- model save path,
- metrics output path.

---

## Roadmap

- Add experiment tracking with MLflow
- Add model registry support
- Add Prometheus/Grafana API monitoring
- Add anomaly detection on forecast residuals
- Add batch inference pipeline
- Add more forecasting models such as GRU, TCN, Transformer, and N-BEATS
- Add Docker Compose for API and monitoring stack
- Add examples for real IoT sensor datasets

---

## Author

**Khojiakbar Botirov**  
GitHub: [@Xojiakbar](https://github.com/Xojiakbar)

---

## License

No license file is currently included. Add a license before public reuse or distribution.

