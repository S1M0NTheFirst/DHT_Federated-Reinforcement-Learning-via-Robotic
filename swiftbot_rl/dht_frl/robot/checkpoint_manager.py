"""
UnifiedCheckpointManager — the core contribution.

Implements Option B: migration is triggered from the DHT orchestrator (host),
not from inside the container.

The "Unified Agent State" = CRIU checkpoint + policy_weights.pt + replay_buffer.pkl
All three transfer in parallel (pipelined) to minimize downtime.

The container signals the host to trigger migration by writing a flag to Redis.
The DHT orchestrator monitors Redis and calls CRIU when the flag appears.
"""
import os
import json
import time
import shutil
import subprocess
import pickle
import torch
import redis
import threading
import logging
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../shared"))
from metrics_collector import get_gpu_util, get_cpu_util, get_net_bytes

logger = logging.getLogger(__name__)


class UnifiedCheckpointManager:

    def __init__(self, robot_id: str, container_name: str,
                 checkpoint_base: str = "/tmp/swiftbot_checkpoints",
                 redis_host: str = "localhost"):
        self.robot_id = robot_id
        self.container_name = container_name
        self.chk_dir = os.path.join(checkpoint_base, robot_id)
        self.r = redis.Redis(host=redis_host, decode_responses=True)
        os.makedirs(self.chk_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # PACK (called by DHT orchestrator after it decides to migrate)
    # ------------------------------------------------------------------ #
    def pack(self, agent, success_rate: float, trigger_reason: str = "load") -> dict:
        """
        Step 1 of migration: save policy + buffer to checkpoint dir.
        CRIU is NOT called here — that happens from the host in the DHT runner.
        Returns timing dict.
        """
        t0 = time.perf_counter()
        gpu_pre = get_gpu_util()
        cpu_pre = get_cpu_util()
        net_pre = get_net_bytes()

        # Save policy weights
        weights_path = os.path.join(self.chk_dir, "policy_weights.pt")
        agent.save_checkpoint(weights_path)

        # Save replay buffer tail
        buffer_path = os.path.join(self.chk_dir, "replay_buffer.pkl")
        tail = agent.replay_buffer.tail(1000)
        with open(buffer_path, "wb") as f:
            pickle.dump(tail, f)

        # Write manifest
        manifest = {
            "robot_id": self.robot_id,
            "container_name": self.container_name,
            "migration_timestamp": time.time(),
            "trigger_reason": trigger_reason,
            "policy_version": agent.training_step,
            "replay_buffer_size": len(agent.replay_buffer),
            "success_rate_premigration": round(success_rate, 4),
            "weights_path": weights_path,
            "buffer_path": buffer_path,
        }
        manifest_path = os.path.join(self.chk_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        t_pack_ms = (time.perf_counter() - t0) * 1000

        # Signal host that policy is saved and ready for CRIU
        self.r.set(f"ready_for_criu:{self.robot_id}", "1", ex=60)

        logger.info(f"[{self.robot_id}] Pack complete in {t_pack_ms:.1f}ms. "
                    f"weights={os.path.getsize(weights_path)/1024:.1f}KB "
                    f"buffer={os.path.getsize(buffer_path)/1024:.1f}KB")

        return {
            "pack_ms": round(t_pack_ms, 2),
            "gpu_pre": gpu_pre,
            "cpu_pre": cpu_pre,
            "net_pre": net_pre,
            "weights_size_kb": os.path.getsize(weights_path) / 1024,
            "buffer_size_kb": os.path.getsize(buffer_path) / 1024,
        }

    # ------------------------------------------------------------------ #
    # RESTORE (called at destination after container is restored)
    # ------------------------------------------------------------------ #
    def restore(self, agent, checkpoint_dir: str) -> dict:
        """
        Step 3 of migration: load policy + replay buffer at destination.
        Container already restored by CRIU at this point.
        """
        t0 = time.perf_counter()

        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load policy weights
        t_pol_start = time.perf_counter()
        weights_path = os.path.join(checkpoint_dir, "policy_weights.pt")
        if os.path.exists(weights_path):
            agent.load_checkpoint(weights_path)
        policy_load_ms = (time.perf_counter() - t_pol_start) * 1000

        # Load replay buffer
        buffer_path = os.path.join(checkpoint_dir, "replay_buffer.pkl")
        entries_restored = 0
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                tail = pickle.load(f)
            agent.replay_buffer.load_tail(tail)
            entries_restored = len(tail)

        total_restore_ms = (time.perf_counter() - t0) * 1000

        logger.info(f"[{self.robot_id}] Restore complete: "
                    f"policy_load={policy_load_ms:.1f}ms "
                    f"buffer={entries_restored} entries")

        return {
            "policy_load_ms": round(policy_load_ms, 2),
            "replay_buffer_entries_restored": entries_restored,
            "total_restore_ms": round(total_restore_ms, 2),
            "success_rate_premigration": manifest.get("success_rate_premigration", 0),
        }
