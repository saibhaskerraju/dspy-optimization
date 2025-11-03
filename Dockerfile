# Default to Python 3.12
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir uv==0.9.7

WORKDIR /app

COPY uv.lock pyproject.toml ./

# Install dependencies
RUN uv sync --frozen --no-cache