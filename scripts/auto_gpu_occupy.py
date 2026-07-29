#!/usr/bin/env python3
"""Dynamically keep selected GPUs busy at a target utilization."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def query_gpu_utils() -> dict[int, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    utils: dict[int, int] = {}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        index, util = [part.strip() for part in line.split(",")]
        utils[int(index)] = int(util)
    return utils


def parse_gpus(value: str) -> list[int]:
    if value == "all":
        return sorted(query_gpu_utils())
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def worker(control_file: str, matrix_size: int, dtype: str) -> None:
    import torch

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch_dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch_dtype)

    print(
        f"worker pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    while True:
        _ = a @ b
        torch.cuda.synchronize()

        try:
            sleep_ms = int(Path(control_file).read_text().strip())
        except Exception:
            sleep_ms = 0

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)


def controller(gpus: list[int], target: int, matrix_size: int, dtype: str, interval: float) -> None:
    control_dir = Path(tempfile.mkdtemp(prefix="gpu_occupy_"))
    workers: dict[int, dict[str, object]] = {}

    print(f"control_dir={control_dir}", flush=True)

    for gpu in gpus:
        control_file = control_dir / f"gpu{gpu}.sleep_ms"
        control_file.write_text("0")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        proc = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--worker",
                "--control-file",
                str(control_file),
                "--matrix-size",
                str(matrix_size),
                "--dtype",
                dtype,
            ],
            env=env,
        )
        workers[gpu] = {"proc": proc, "control_file": control_file, "sleep_ms": 0}
        print(f"GPU {gpu}: started pid={proc.pid}", flush=True)

    def stop_workers(signum: int | None = None, frame: object | None = None) -> None:
        print("stopping workers...", flush=True)
        for item in workers.values():
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            if proc.poll() is None:
                proc.terminate()
        time.sleep(2)
        for item in workers.values():
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            if proc.poll() is None:
                proc.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)

    while True:
        utils = query_gpu_utils()
        for gpu, item in workers.items():
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            control_file = item["control_file"]
            assert isinstance(control_file, Path)
            sleep_ms = int(item["sleep_ms"])

            if proc.poll() is not None:
                print(f"GPU {gpu}: worker exited with code {proc.returncode}", flush=True)
                continue

            util = utils.get(gpu, 0)
            if util < target:
                sleep_ms = max(0, sleep_ms - 10)
            elif util > min(99, target + 5):
                sleep_ms = min(250, sleep_ms + 5)

            item["sleep_ms"] = sleep_ms
            control_file.write_text(str(sleep_ms))
            print(f"GPU {gpu}: util={util:3d}% sleep={sleep_ms:3d}ms pid={proc.pid}", flush=True)

        print("-" * 48, flush=True)
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="all", help="'all' or a comma list, e.g. 0,1,3")
    parser.add_argument("--target", type=int, default=90)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--control-file")
    args = parser.parse_args()

    if args.worker:
        if not args.control_file:
            raise ValueError("--control-file is required in worker mode")
        worker(args.control_file, args.matrix_size, args.dtype)
        return

    controller(parse_gpus(args.gpus), args.target, args.matrix_size, args.dtype, args.interval)


if __name__ == "__main__":
    main()
