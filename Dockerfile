FROM python:3.11-slim

# Runtime BLAS/LAPACK for NumPy/SciPy wheels + R for ADMIXTOOLS 2.
# GSL: bookworm has libgsl27, trixie+ has libgsl28 — install whichever exists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0-pthread \
    libopenblas-dev \
    liblapack3 \
    liblapack-dev \
    libgfortran5 \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    git \
    && (apt-get install -y --no-install-recommends libgsl27 \
        || apt-get install -y --no-install-recommends libgsl28) \
    && ldconfig \
    && test -n "$(find /usr/lib -name libopenblas.so.0 -print -quit)" \
    && rm -rf /var/lib/apt/lists/*

# Install ADMIXTOOLS 2 R package + jsonlite for structured output.
# Pin readr to 2.1.5: newer versions removed read_table2() which admixtools still calls.
RUN R -e "install.packages(c('remotes', 'jsonlite'), repos='https://cloud.r-project.org')" \
    && R -e "remotes::install_version('readr', version='2.1.5', repos='https://cloud.r-project.org')" \
    && R -e "remotes::install_github('uqrmaie1/admixtools', upgrade='never')"

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn numpy scipy pandas pydantic python-multipart python-dotenv

# Prevent BLAS/OpenMP thread contention when running multiple concurrent conversions.
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

COPY main.py admix_models.py admix_fraction.py raw_data_processing.py progress_tracker.py k36_to_g25_weights.csv ./
COPY app/ app/
COPY admixture_jobs/ admixture_jobs/
COPY qpadm/ qpadm/
COPY scripts/ scripts/
COPY data/ data/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
