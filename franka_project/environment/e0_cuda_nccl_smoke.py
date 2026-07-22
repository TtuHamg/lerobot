#!/usr/bin/env python3
"""Minimal CUDA/cuDNN/BF16/NCCL environment smoke for the Frank3 project."""

from __future__ import annotations

import json
import os
import socket
from multiprocessing import get_context

import torch


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _nccl_worker(rank: int, world_size: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    torch.distributed.all_reduce(value)
    queue.put((rank, float(value.item())))
    torch.distributed.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This smoke requires the two GPUs planned for training")

    convolution_results = []
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        layer = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device=device, dtype=torch.bfloat16)
        sample = torch.randn(2, 3, 32, 32, device=device, dtype=torch.bfloat16)
        output = layer(sample)
        loss = output.float().square().mean()
        loss.backward()
        convolution_results.append(
            {
                "device": index,
                "name": torch.cuda.get_device_name(index),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "gradient_finite": bool(torch.isfinite(layer.weight.grad).all().item()),
            }
        )

    context = get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_nccl_worker, args=(rank, 2, port, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"NCCL worker exit codes: {[process.exitcode for process in processes]}")
    reduced = dict(queue.get(timeout=5) for _ in range(2))
    if reduced != {0: 3.0, 1: 3.0}:
        raise RuntimeError(f"Unexpected NCCL all-reduce values: {reduced}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "nccl_version": torch.cuda.nccl.version(),
                "bf16_convolution": convolution_results,
                "nccl_all_reduce": reduced,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
