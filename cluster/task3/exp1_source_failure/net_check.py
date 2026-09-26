"""Exp1 network preflight (copy of task3/exp2_dht_rendezvous/net_check.py): can
the job's nodes reach each other on the ports the DHT ring uses (UDP) and the
control/Redis ports (TCP)?

  serve : echo UDP datagrams on --udp-ports and accept TCP on --tcp-port, then
          exit after --seconds (self-terminating, nothing is left behind).
  probe : from this node, check every --target; exit 1 if a required port fails.

  python3 net_check.py serve --udp-ports 23100,23107 --tcp-port 8790 --seconds 90
  python3 net_check.py probe --udp-ports 23100,23107 --udp-info 23000 \
      --target n017.cluster:8790,6779 --target n025.cluster:8790
"""
import argparse
import os
import socket
import sys
import threading
import time


def ports(s: str) -> list[int]:
    return [int(p) for p in s.split(",") if p.strip()]


def serve(a: argparse.Namespace) -> int:
    socks = []
    try:
        for p in ports(a.udp_ports) + ports(a.udp_info):
            u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            u.bind(("0.0.0.0", p))
            socks.append(u)
        t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        t.bind(("0.0.0.0", a.tcp_port))
        t.listen(16)
    except OSError as e:
        print(f"net_check serve FAILED to bind on {socket.gethostname()}: {e!r}",
              flush=True)
        return 1

    def udp_echo(u: socket.socket) -> None:
        while True:
            data, addr = u.recvfrom(512)
            u.sendto(data, addr)

    def tcp_accept() -> None:
        while True:
            c, _ = t.accept()
            c.close()

    for u in socks:
        threading.Thread(target=udp_echo, args=(u,), daemon=True).start()
    threading.Thread(target=tcp_accept, daemon=True).start()
    print(f"net_check serving on {socket.gethostname()}: udp "
          f"{a.udp_ports},{a.udp_info} tcp {a.tcp_port}", flush=True)
    # Stop at --seconds, or as soon as the job's SSH session / shell is gone.
    parent, end = os.getppid(), time.time() + a.seconds
    while time.time() < end and os.getppid() == parent:
        time.sleep(1.0)
    return 0


def udp_ok(host: str, port: int, tries: int = 5) -> bool:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.settimeout(1.0)
    try:
        for i in range(tries):
            try:
                u.sendto(f"ping{i}".encode(), (host, port))
                data, _ = u.recvfrom(512)
                if data.startswith(b"ping"):
                    return True
            except OSError:
                pass
        return False
    finally:
        u.close()


def tcp_ok(host: str, port: int) -> bool:
    try:
        socket.create_connection((host, port), timeout=5).close()
        return True
    except OSError:
        return False


def probe(a: argparse.Namespace) -> int:
    me = socket.gethostname().split(".")[0]
    bad = 0
    for spec in a.target:
        host, _, tcp = spec.partition(":")
        short = host.split(".")[0]
        checks = ([("udp", p, True) for p in ports(a.udp_ports)]
                  + [("udp", p, False) for p in ports(a.udp_info)]
                  + [("tcp", p, True) for p in ports(tcp)])
        for proto, p, required in checks:
            ok = (udp_ok(host, p, 5 if required else 2) if proto == "udp"
                  else tcp_ok(host, p))
            tag = "OK" if ok else ("BLOCKED" if required else "blocked (info only)")
            print(f"NET {me} -> {short} {proto}/{p}: {tag}", flush=True)
            if required and not ok:
                bad += 1
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["serve", "probe"])
    p.add_argument("--udp-ports", default="", help="required UDP ports")
    p.add_argument("--udp-info", default="", help="extra UDP ports, reported only")
    p.add_argument("--tcp-port", type=int, default=0, help="serve: TCP port")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--target", action="append", default=[],
                   help="probe: HOST[:TCP,TCP...]; repeat per node")
    a = p.parse_args()
    return serve(a) if a.mode == "serve" else probe(a)


if __name__ == "__main__":
    sys.exit(main())
