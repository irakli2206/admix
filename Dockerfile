FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn numpy scipy pandas pydantic python-multipart

# Prevent BLAS/OpenMP thread contention when running multiple concurrent conversions.
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

COPY main.py admix_models.py admix_fraction.py raw_data_processing.py progress_tracker.py k36_to_g25_weights.csv ./
COPY app/ app/
COPY qpadm/ qpadm/
COPY data/ data/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
