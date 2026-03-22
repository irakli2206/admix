# Cheapest reliable deploy: run on a $5–6 VPS (e.g. Contabo, Hetzner). 2+ GB RAM.
FROM python:3.11-slim

WORKDIR /app

# App deps (no git dependency for API)
RUN pip install --no-cache-dir \
    fastapi uvicorn numpy scipy pandas pydantic python-multipart

COPY main.py admix_models.py admix_fraction.py raw_data_processing.py k36_to_g25_weights.csv ./
COPY data/ data/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
