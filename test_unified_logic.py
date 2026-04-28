import os
import sys
import time
import shutil
import subprocess
import docker
import logging

# Add shared dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "swiftbot_rl/shared"))
from metrics_collector import real_criu_dump, cuda_checkpoint_toggle, get_container_pid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFIER")

def verify_sudo():
    print("--- Checking Non-Interactive Sudo ---")
    cmds = [
        ["sudo", "-n", "criu", "--version"],
        ["sudo", "-n", "/usr/local/bin/cuda-checkpoint", "--help"],
        ["sudo", "-n", "chown", "--version"]
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  OK: {' '.join(cmd)}")
        except Exception as e:
            print(f"  FAILED: {' '.join(cmd)} - {e}")
            return False
    return True

def run_migration_sequence(pid, run_id):
    print(f"\n--- Running Migration Sequence (Round {run_id}) ---")
    
    chk_src = f"/tmp/verifier_chk/robot_src"
    chk_dst = f"/tmp/verifier_chk/robot_dst"
    
    criu_dir = os.path.join(chk_src, "criu")
    
    # Exact logic from runner
    if os.path.exists(criu_dir):
        shutil.rmtree(criu_dir)
    if os.path.exists(chk_dst):
        shutil.rmtree(chk_dst)
        
    os.makedirs(criu_dir, exist_ok=True)
    os.makedirs(chk_dst, exist_ok=True)

    print("  Toggling CUDA OFF...")
    if not cuda_checkpoint_toggle(pid):
        print("  FAILED to toggle CUDA")
        return False

    parent = ""
    for i in range(3):
        predump_dir = os.path.join(criu_dir, f"predump_{i}")
        print(f"  Pre-dump {i} (parent={parent})...")
        res = real_criu_dump(pid, predump_dir, parent_dir=parent, pre_dump=True)
        if res["returncode"] != 0:
            print(f"  FAILED pre-dump {i}: {res['stderr']}")
            return False
        parent = predump_dir
        time.sleep(0.05)

    print("  Final dump...")
    final_dir = os.path.join(criu_dir, "final")
    res = real_criu_dump(pid, final_dir, parent_dir=parent, pre_dump=False)
    if res["returncode"] != 0:
        print(f"  FAILED final dump: {res['stderr']}")
        return False

    print("  Testing symlinked transfer...")
    try:
        t0 = time.time()
        dst_criu = os.path.join(chk_dst, "criu")
        if os.path.exists(dst_criu):
            shutil.rmtree(dst_criu)
        shutil.copytree(criu_dir, dst_criu, symlinks=True)
        print(f"  Transfer SUCCESS in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  Transfer FAILED: {e}")
        return False

    print("  Toggling CUDA back ON...")
    if not cuda_checkpoint_toggle(pid):
        print("  FAILED to toggle CUDA back ON")
        return False

    return True

def test_unified_sequence():
    if not verify_sudo():
        return

    client = docker.from_env()
    cname = "test-unified-fix-final"
    try:
        client.containers.get(cname).remove(force=True)
    except:
        pass
        
    print("\n  Launching CUDA container...")
    container = client.containers.run(
        "swiftbot-robot:latest",
        command="python3 -c 'import torch; x=torch.randn(10,10,device=\"cuda\"); import time; [time.sleep(1) for _ in range(100)]'",
        name=cname,
        detach=True,
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        security_opt=["seccomp:unconfined"]
    )
    
    time.sleep(5) 
    pid = get_container_pid(cname)
    print(f"  Container PID: {pid}")

    subprocess.run(["sudo", "-n", "rm", "-rf", "/tmp/verifier_chk"])

    # Run it twice to simulate multiple migrations (which causes Round 11 issues)
    success1 = run_migration_sequence(pid, 1)
    if not success1:
        print("[FAIL] First migration failed.")
        container.remove(force=True)
        return
        
    time.sleep(2)
    
    success2 = run_migration_sequence(pid, 2)
    if not success2:
        print("[FAIL] Second migration failed (The Round 11 issue!).")
        container.remove(force=True)
        return

    print("\n[SUCCESS] Environment is 100% verified.")
    print("Multiple migrations can occur cleanly without pollution or symlink errors.")
    container.remove(force=True)

if __name__ == "__main__":
    test_unified_sequence()
