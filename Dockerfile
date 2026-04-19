# ----- builder stage -----
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch/torchvision from the PyTorch CPU index first;
# the unpinned `torch`/`torchvision` lines in requirements.txt then resolve as already-satisfied.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision==0.26.0 && \
    pip install --no-cache-dir -r requirements.txt

# Bake FaceNet weights into the image cache at build time.
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2').eval()"

# ----- runtime stage -----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

ENV PYTHONPATH=/app

CMD ["python", "scripts/verify.py", "--help"]
