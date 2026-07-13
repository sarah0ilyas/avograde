FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" python-multipart

COPY avograder_resnet.pt ./avograder_resnet.pt

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn avograde.serving.app:build_app --factory --host 0.0.0.0 --port ${PORT}"]
