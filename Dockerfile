FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_updated.txt .

RUN pip install --no-cache-dir -r requirements_updated.txt

COPY . .

RUN mkdir -p checkpoints static temp

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN echo '#!/bin/bash\n\
echo "Starting Real-Time Lip-Syncing WebSocket API in Docker..."\n\
echo "Checking for models in /app/checkpoints/..."\n\
ls -la /app/checkpoints/ || echo "Checkpoints directory empty - models should be mounted or downloaded"\n\
echo "Starting server..."\n\
exec python -m app.main' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]