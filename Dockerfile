FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INPUT_FILE=/input/tasks.json \
    OUTPUT_FILE=/output/results.json \
    LOCAL_MODEL_ENABLED=true \
    LOCAL_PREFERRED_DEVICE=CPU \
    LOCAL_FALLBACK_DEVICE=CPU \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-int4-ov \
    FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /input /output

CMD ["python", "docker_entrypoint.py"]
