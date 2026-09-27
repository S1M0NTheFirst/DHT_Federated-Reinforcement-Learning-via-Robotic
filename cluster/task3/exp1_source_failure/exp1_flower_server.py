"""
Exp1 Flower server = task2's server (imported unchanged from
/cluster_app/task2/flower_server.py: seeding, metrics, fl_*.csv, task_logs.csv)
with two additions that a fleet whose migration sources get SIGKILLed needs:

  - ServerConfig(round_timeout=EXP1_ROUND_TIMEOUT): a client that never answers
    costs at most that long (flwr 1.5 then aborts its stream, which closes its
    bridge and unregisters it), instead of blocking the round indefinitely.
  - ghost pruning: clients add "robot_id" to their fit/evaluate metrics, so the
    server knows which proxy is which robot and when it last answered. The
    runner records every kill in Redis (exp1_killed:<robot> = unix time). A
    proxy whose robot was killed AFTER that proxy's last result is a dead
    source: its bridge is closed (a pending fit/evaluate fails at once) and it
    is unregistered. Proxies that have not answered yet (e.g. the relaunched
    robot) are never touched; min_available_clients stays N, so the next round
    waits for the relaunched robot.

Env (besides task2's): EXP1_ROUND_TIMEOUT (s, 0 = none).
"""
import logging
import os
import sys
import threading
import time

TASK2 = os.environ.get("TASK2_DIR", "/cluster_app/task2")
sys.path.insert(0, TASK2)
import flower_server as t2  # noqa: E402  (task2 server module, unchanged)
import flwr as fl  # noqa: E402

logger = logging.getLogger("exp1_server")
ROUND_TIMEOUT = float(os.environ.get("EXP1_ROUND_TIMEOUT", "150")) or None


class Exp1FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, redis_client=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = redis_client
        self.robot_of: dict = {}      # proxy cid -> robot_id
        self.last_seen: dict = {}     # proxy cid -> unix time of its last result
        self.lock = threading.Lock()
        self.cm = None
        self.dropped = 0

    def note_results(self, results) -> None:
        now = time.time()
        with self.lock:
            for proxy, res in results:
                rid = (getattr(res, "metrics", None) or {}).get("robot_id")
                if rid:
                    self.robot_of[proxy.cid] = str(rid)
                    self.last_seen[proxy.cid] = now

    def prune(self, cm) -> int:
        """Drop proxies of robots killed after the proxy's last result."""
        if cm is None or self.r is None:
            return 0
        n = 0
        for cid, proxy in list(cm.all().items()):
            with self.lock:
                rid, seen = self.robot_of.get(cid), self.last_seen.get(cid)
            if not rid or seen is None:
                continue                      # never answered: never touched
            try:
                raw = self.r.get(f"exp1_killed:{rid}")
            except Exception as e:            # noqa: BLE001
                logger.warning("ghost check: redis error %r", e)
                return n
            if raw is None or float(raw) <= seen:
                continue
            logger.warning("dropping ghost client %s (%s killed %.1fs after its "
                           "last result)", cid, rid, float(raw) - seen)
            bridge = getattr(proxy, "bridge", None)
            if bridge is not None:
                try:
                    bridge.close()            # a pending request fails at once
                except Exception:             # noqa: BLE001
                    pass
            cm.unregister(proxy)
            with self.lock:
                self.robot_of.pop(cid, None)
                self.last_seen.pop(cid, None)
            self.dropped += 1
            n += 1
        return n

    def watch(self, period: float = 1.0) -> None:
        while True:
            try:
                self.prune(self.cm)
            except Exception as e:            # noqa: BLE001
                logger.warning("ghost watcher: %r", e)
            time.sleep(period)

    def configure_fit(self, server_round, parameters, client_manager):
        self.cm = client_manager
        self.prune(client_manager)
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        self.cm = client_manager
        self.prune(client_manager)
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        self.note_results(results)
        if failures:
            logger.info("round %d fit: %d results, %d failures", server_round,
                        len(results), len(failures))
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        self.note_results(results)
        return super().aggregate_evaluate(server_round, results, failures)


def run():
    import redis
    params = fl.common.ndarrays_to_parameters(t2._initial_actor_arrays())

    def config_fn(server_round: int):
        # task2's weighted_average reads these module globals for lat=/net=.
        t2.round_start_time = time.time()
        net = t2.psutil.net_io_counters()
        t2.round_start_net = net.bytes_sent + net.bytes_recv
        return {"round": server_round}

    strategy = Exp1FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=t2.N_CLIENTS, min_evaluate_clients=t2.N_CLIENTS,
        min_available_clients=t2.N_CLIENTS,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=t2.weighted_average,
        fit_metrics_aggregation_fn=t2.weighted_average,
        on_fit_config_fn=config_fn,
        on_evaluate_config_fn=config_fn,
        redis_client=redis.Redis(host=t2.REDIS_HOST, port=t2.REDIS_PORT,
                                 decode_responses=True),
    )
    threading.Thread(target=strategy.watch, daemon=True).start()
    logger.info("Exp1 server: %d clients x %d rounds, round_timeout=%s, "
                "ghost pruning on", t2.N_CLIENTS, t2.N_ROUNDS, ROUND_TIMEOUT)
    hist = fl.server.start_server(
        server_address=t2.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=t2.N_ROUNDS,
                                      round_timeout=ROUND_TIMEOUT),
        strategy=strategy,
    )
    logger.info("ghost clients dropped: %d", strategy.dropped)
    return hist


def main():
    os.makedirs(t2.RESULT_DIR, exist_ok=True)
    time.sleep(15)
    try:
        t2.save_results(run())
    finally:
        t2.persist_task_logs()
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    main()
