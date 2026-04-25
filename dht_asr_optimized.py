# dht_asr_optimized.py
import asyncio
import docker
import os
import time
import sys
import platform
from kademlia.network import Server
import logging

# --- CONFIG ---
RAW_DATA_PATH = r"C:\Users\Simon\Desktop\Summer\LibriSpeech\train-clean-100" 
DOCKER_IMAGE_NAME = "asr-app:optimized" 
NUM_NODES = 4
CLIENTS_PER_NODE = 2
TOTAL_CLIENTS = NUM_NODES * CLIENTS_PER_NODE
BASE_PORT = 8470

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class DHTNode:
    def __init__(self, node_id, port, bootstrap=None):
        self.node_id = node_id
        self.port = port
        self.bootstrap = bootstrap
        self.server = Server()
        self.docker = docker.from_env()

    async def start(self, master_ip):
        await self.server.listen(self.port)
        if self.bootstrap: await self.server.bootstrap([self.bootstrap])
        return await self.launch_workers(master_ip)

    async def launch_workers(self, master_ip):
        print(f"[Node {self.node_id}] Checking containers...")
        for i in range(CLIENTS_PER_NODE):
            cid = (self.node_id * CLIENTS_PER_NODE) + i
            cname = f"asr-loop-client-{cid}"
            
            try:
                # Remove if exists to ensure fresh start with correct IP
                try:
                    c = self.docker.containers.get(cname)
                    if c.status in ['exited', 'dead', 'created']:
                        c.remove(force=True)
                    elif c.status == 'running':
                        # Check if environment var matches new master IP
                        env = c.attrs['Config']['Env']
                        current_master = next((e for e in env if "MASTER_ADDRESS" in e), "")
                        if master_ip not in current_master:
                            logger.info(f"Recreating {cname} with new Master IP...")
                            c.remove(force=True)
                        else:
                            continue # Already running correctly
                except docker.errors.NotFound:
                    pass

                cmd = f"python3 /app/worker_client_asr_optimized.py --client-id {cid} --num-clients {TOTAL_CLIENTS}"
                self.docker.containers.run(
                    DOCKER_IMAGE_NAME, command=cmd, name=cname, detach=True, tty=True,
                    shm_size='8g',
                    environment={
                        "MASTER_ADDRESS": f"{master_ip}:8080",
                        "NVIDIA_VISIBLE_DEVICES": "all",
                        "PYTHONUNBUFFERED": "1",
                        "GLOG_minloglevel": "2",
                        "GRPC_VERBOSITY": "ERROR"
                    },
                    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                    volumes={RAW_DATA_PATH: {"bind": "/app/data/LibriSpeech/train-clean-100/", "mode": "ro"}},
                    restart_policy={"Name": "on-failure", "MaximumRetryCount": 5}
                )
                logger.info(f"Started {cname}")
            except Exception as e:
                logger.error(f"Error launching {cname}: {e}")
            await asyncio.sleep(0.5)

async def main():
    if not os.path.exists(RAW_DATA_PATH): return print(f"ERROR: {RAW_DATA_PATH} not found")
    
    # --- NETWORK FIX ---
    # Force host.docker.internal for Windows Docker Desktop
    if sys.platform == "win32" or "microsoft" in platform.uname().release.lower():
        master_ip = "host.docker.internal"
        logger.info(f"Detected Windows/WSL. Using Docker magic DNS: {master_ip}")
    else:
        # Fallback for Linux
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        master_ip = s.getsockname()[0]
        s.close()
        logger.info(f"Detected Linux IP: {master_ip}")

    # Create Nodes
    nodes = [DHTNode(0, BASE_PORT)] + [DHTNode(i, BASE_PORT+i, ("127.0.0.1", BASE_PORT)) for i in range(1, NUM_NODES)]
    
    print(f"Launching {TOTAL_CLIENTS} clients pointing to {master_ip}:8080...")
    await asyncio.gather(*[n.start(master_ip) for n in nodes])
    
    print("\n[SUCCESS] Clients running.")
    print("1. Clients are now waiting for server connection.")
    print("2. Run 'python server_asr_optimized.py' to start the experiment.")
    
    try:
        while True: await asyncio.sleep(10)
    except KeyboardInterrupt: print("Stopped.")

if __name__ == "__main__":
    asyncio.run(main())