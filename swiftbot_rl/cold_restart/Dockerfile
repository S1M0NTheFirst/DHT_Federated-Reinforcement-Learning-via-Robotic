# Cold-restart baseline image. Same worker, no CRIU needed (we just kill +
# relaunch). Build:
#   docker build -f cold_restart/Dockerfile -t swiftbot-cold-restart:latest .

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV GRPC_VERBOSITY=ERROR
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

RUN pip3 install --no-cache-dir torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    psutil==5.9.8 pynvml==11.5.0 redis==5.0.1 \
    numpy==1.24.4 pandas==2.0.3 matplotlib==3.7.2

COPY cold_restart/worker_random_client.py /app/worker_random_client.py
COPY dht_frl/robot/ /app/robot/

RUN mkdir -p /app/results

CMD ["python3", "--version"]
