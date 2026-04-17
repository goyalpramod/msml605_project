FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2').eval()"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/

ENV PYTHONPATH=/app

CMD ["python", "scripts/verify.py", "--help"]
