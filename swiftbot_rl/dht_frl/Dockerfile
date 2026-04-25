# SwiftBot-RL robot container — DHT+FRL condition
# Based on Dockerfile.optimized — REMOVE torchaudio/jiwer/libsndfile/ffmpeg/sox
# ADD criu, stable-baselines3, gymnasium, pynvml, redis

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV GRPC_VERBOSITY=ERROR
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# System packages — ADD criu, REMOVE libsndfile/ffmpeg/sox
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    criu \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch (no torchaudio)
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Robot dependencies — REMOVE jiwer/torchaudio, ADD new packages
RUN pip3 install --no-cache-dir \
    flwr==1.5.0 \
    psutil==5.9.8 \
    pynvml==11.5.0 \
    redis==5.0.1 \
    numpy==1.24.4 \
    stable-baselines3==2.2.1 \
    gymnasium==0.29.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2

# Copy robot code
COPY worker_robot_client.py /app/
COPY robot/ /app/robot/

# Create checkpoint directory (host mounts /tmp/swiftbot_checkpoints here)
RUN mkdir -p /checkpoints /app/results

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD pgrep -f python3 || exit 1

CMD ["python3", "--version"]
