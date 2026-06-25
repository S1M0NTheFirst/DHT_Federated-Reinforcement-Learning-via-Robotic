# Motivation experiment — D4RL hopper-medium-v2 offline RL agent.
# Extends the swiftbot-baseline image (same torch/CUDA stack used by the
# criu_cold baseline) so the CRIU dump size is directly comparable. The
# only delta vs. swiftbot-baseline is the minari[hf] install for loading
# `mujoco/hopper/medium-v0` (Farama's port of d4rl hopper-medium-v2).
#
# Build from swiftbot_rl/ as context (after swiftbot-baseline is built):
#   docker build -f motivation/Dockerfile -t swiftbot-motivation:latest .
FROM swiftbot-baseline:latest

RUN pip3 install --no-cache-dir "minari[hf]==0.5.3" "huggingface_hub>=0.23"

COPY motivation/hopper_agent.py /app/hopper_agent.py

CMD ["python3", "/app/hopper_agent.py", \
     "--steps",      "500", \
     "--batch",      "256", \
     "--save-path",  "/checkpoints/hopper_policy.pt", \
     "--ready-path", "/checkpoints/hopper_ready.json"]
