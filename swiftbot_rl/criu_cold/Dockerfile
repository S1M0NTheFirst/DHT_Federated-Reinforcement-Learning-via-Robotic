# SwiftBot-RL baseline container — CRIU Cold and CRIU Warm conditions
# Build from swiftbot_rl/ as context:
#   docker build -f criu_cold/Dockerfile -t swiftbot-baseline:latest .

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV GRPC_VERBOSITY=ERROR
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    criu \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    psutil==5.9.8 \
    pynvml==11.5.0 \
    redis==5.0.1 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    matplotlib==3.7.2

# Copy baseline worker (same file used for both cold and warm conditions)
COPY criu_cold/worker_random_client.py /app/worker_random_client.py

# Copy robot modules (task_generator needed by baseline worker)
COPY dht_frl/robot/ /app/robot/

RUN mkdir -p /checkpoints /app/results

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD pgrep -f python3 || exit 1

CMD ["python3", "--version"]
