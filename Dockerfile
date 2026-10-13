FROM python:3.13-slim

# The project venv is first on PATH; src is importable without installing the project.
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    HF_HOME="/models"

WORKDIR /app

# git is needed to install TimesFM from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.23

# 1) Core dependencies exactly as locked in uv.lock (no dev tools in the image).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Foundation-model dependencies live only in the image, never in uv.lock:
#    PyTorch has no wheels for Intel Macs and would break `uv lock` on the host.
#    CPU-only PyTorch: the default Linux wheel pulls CUDA libraries we cannot use here.
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install "chronos-forecasting==2.3.2" "pandas[pyarrow]"

# TimesFM 3.0 is installed from GitHub and pinned to a commit for reproducibility.
ARG TIMESFM_REF=e51928e27119cb17bebc005be2696b75e0a9e688
RUN test -n "$TIMESFM_REF" \
    && uv pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@${TIMESFM_REF}"

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src ./src
COPY scripts ./scripts

CMD ["python", "scripts/fm_smoke_test.py"]
