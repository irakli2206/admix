FROM python:3.11-slim

# Runtime libs for host-built qpAdm (and similar) mounted into the container.
# On trixie, libopenblas0 is a metapackage; pull pthread impl so libopenblas.so.0 is on disk.
# GSL: bookworm has libgsl27, trixie+ has libgsl28 — install whichever exists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0-pthread \
    liblapack3 \
    libgfortran5 \
    && (apt-get install -y --no-install-recommends libgsl27 \
        || apt-get install -y --no-install-recommends libgsl28) \
    && ldconfig \
    && test -n "$(find /usr/lib -name libopenblas.so.0 -print -quit)" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn numpy scipy pandas pydantic python-multipart python-dotenv

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
