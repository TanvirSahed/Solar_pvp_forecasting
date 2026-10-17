# GPU research image for pvf. Host needs only an NVIDIA driver (>= R580 for CUDA 13.0 wheels) and the
# NVIDIA Container Toolkit (Linux) or Docker Desktop + WSL2 (Windows). CUDA runtime ships inside the torch wheel.
#   docker compose build
#   docker compose run --rm pvf python -m pvf smoke
#   docker compose up lab        # JupyterLab on http://localhost:8888
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements/ requirements/
# requirements/lock-linux.txt is generated inside the container on first build (see README) - if present,
# it pins every package to the exact versions the team validated.
RUN pip install -r requirements/torch-cu130.txt \
 && if [ -f requirements/lock-linux.txt ]; then pip install -r requirements/lock-linux.txt; \
    else pip install -r requirements/core.txt -r requirements/fm.txt; fi

COPY pyproject.toml ./
COPY pvf/ pvf/
COPY configs/ configs/
COPY tools/ tools/
RUN pip install --no-deps -e .

# Data, runs, results and refs are bind-mounted (see docker-compose.yml), not baked into the image.
CMD ["python", "-m", "pvf", "list"]
