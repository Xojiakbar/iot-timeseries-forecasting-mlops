FROM python:3.11-slim



WORKDIR /app



# Install build deps (minimal)

RUN pip install --no-cache-dir --upgrade pip



# Copy project files

COPY pyproject.toml /app/pyproject.toml

COPY src /app/src



# Install runtime deps

RUN pip install --no-cache-dir .



# Copy model optionally (in real deployment, mount models/)

COPY models /app/models



ENV MODEL_PATH=/app/models/model.pt

ENV DEVICE=cpu



EXPOSE 8000

CMD ["uvicorn", "ts_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]


