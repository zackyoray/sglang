"""Two-process NIXL/Mooncake embedding correctness smoke and microbenchmark.

Run receiver first, then sender (same command except --role and --gpu-id).
Use --host to give the receiver's reachable control address on both hosts.
"""

import argparse
import importlib.util
import json
import socket
import statistics
import struct
import time
from pathlib import Path


def send_message(sock, message):
    payload = json.dumps(message).encode()
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(sock):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(size - len(result))
            if not chunk:
                raise EOFError("Peer disconnected")
            result.extend(chunk)
        return result

    size = struct.unpack("!I", exact(4))[0]
    if size > 64 * 1024 * 1024:
        raise ValueError("Control message too large")
    return json.loads(exact(size))


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percent / 100)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--backend", choices=["nixl", "mooncake"], default="nixl")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver control address")
    parser.add_argument(
        "--local-host", default=None, help="Mooncake local RDMA address"
    )
    parser.add_argument("--port", type=int, default=29590)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="NIXL API smoke only")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4096, 1048576, 16777216, 67108864]
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--ib-device", default=None, help="Mooncake NIC selection")
    parser.add_argument("--output", help="Sender JSON output path")
    args = parser.parse_args()
    if min(args.sizes) <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("Sizes/iterations must be positive, warmup nonnegative")
    if args.cpu and args.backend != "nixl":
        parser.error("--cpu is supported only for the NIXL smoke test")

    import torch

    if not args.cpu:
        torch.cuda.set_device(args.gpu_id)
    device = "cpu" if args.cpu else f"cuda:{args.gpu_id}"

    def sync():
        if not args.cpu:
            torch.cuda.synchronize(args.gpu_id)

    if args.backend == "nixl":
        path = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/disaggregation/encoder/transfer.py"
        )
        spec = importlib.util.spec_from_file_location("encoder_transfer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.NixlEmbeddingTransferEngine(
            args.gpu_id,
            memory_type="DRAM" if args.cpu else "VRAM",
            timeout=args.timeout,
            initiator=args.role == "sender",
        )
    else:
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )

        if not args.local_host:
            parser.error(
                "Mooncake requires --local-host with this process reachable address"
            )
        engine = MooncakeTransferEngine(args.local_host, args.gpu_id, args.ib_device)

    # Exercise subranges of a persistent pool, including nonzero byte offsets.
    offset = 256
    buffer = torch.empty(max(args.sizes) + 2 * offset, dtype=torch.uint8, device=device)
    sync()
    started = time.perf_counter()
    engine.register(buffer.data_ptr(), buffer.nbytes)
    registration_ms = (time.perf_counter() - started) * 1000
    results = []
    listener = None
    try:
        if args.role == "receiver":
            listener = socket.create_server((args.host, args.port))
            listener.settimeout(args.timeout)
            print(f"Listening on {args.host}:{args.port}", flush=True)
            sock, _ = listener.accept()
        else:
            sock = socket.create_connection(
                (args.host, args.port), timeout=args.timeout
            )
        with sock:
            sock.settimeout(args.timeout)
            if args.role == "receiver":
                send_message(
                    sock,
                    {
                        "session": engine.session_id,
                        "address": buffer.data_ptr() + offset,
                        "capacity": max(args.sizes),
                        "backend": args.backend,
                    },
                )
                while True:
                    msg = receive_message(sock)
                    if msg.get("done"):
                        break
                    size, value = msg["size"], msg["value"]
                    if size > max(args.sizes):
                        raise ValueError("Sender exceeds receiver capacity")
                    sync()
                    ok = bool(torch.all(buffer[offset : offset + size] == value).item())
                    send_message(sock, {"ok": ok})
                    if not ok:
                        raise AssertionError(
                            f"Embedding mismatch: {size} bytes, value={value}"
                        )
                return
            remote = receive_message(sock)
            if remote["backend"] != args.backend or remote["capacity"] < max(
                args.sizes
            ):
                raise ValueError("Receiver backend/capacity mismatch")
            for size in args.sizes:
                samples = []
                for iteration in range(args.warmup + args.iterations):
                    value = iteration % 251
                    buffer[offset : offset + size].fill_(value)
                    sync()
                    started = time.perf_counter()
                    transfer_kwargs = (
                        {"source_owner": buffer} if args.backend == "nixl" else {}
                    )
                    ret = engine.transfer_sync(
                        remote["session"],
                        buffer.data_ptr() + offset,
                        remote["address"],
                        size,
                        **transfer_kwargs,
                    )
                    elapsed = time.perf_counter() - started
                    if ret != 0:
                        raise RuntimeError(f"Transfer returned {ret}")
                    send_message(sock, {"size": size, "value": value})
                    if not receive_message(sock)["ok"]:
                        raise AssertionError("Receiver reported corrupted data")
                    if iteration >= args.warmup:
                        samples.append(elapsed)
                row = {
                    "bytes": size,
                    "iterations": len(samples),
                    "p50_ms": percentile(samples, 50) * 1000,
                    "p95_ms": percentile(samples, 95) * 1000,
                    "p99_ms": percentile(samples, 99) * 1000,
                    "mean_GB_s": size / statistics.mean(samples) / 1e9,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
            send_message(sock, {"done": True})
    finally:
        engine.deregister(buffer.data_ptr())
        if listener:
            listener.close()
    report = {
        "backend": args.backend,
        "device": device,
        "registration_ms": registration_ms,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
