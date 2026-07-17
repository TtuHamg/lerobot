#!/usr/bin/env python
"""Serve the project Franka PI0 checkpoint through LeRobot async inference."""

import logging
import sys
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import grpc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from franka_eef_pipeline.async_server import FrankaPI0PolicyServer  # noqa: E402
from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.transport import services_pb2_grpc  # noqa: E402


def validate_phase1_config(cfg: PolicyServerConfig) -> None:
    """Fail fast on settings that would violate the approved dry-run contract."""

    expected = {
        "host": "127.0.0.1",
        "port": 15173,
        "fps": 15,
        "policy_type": "pi0",
        "actions_per_chunk": 50,
    }
    actual = {name: getattr(cfg, name) for name in expected}
    mismatches = {
        name: {"expected": value, "actual": actual[name]}
        for name, value in expected.items()
        if actual[name] != value
    }
    if mismatches:
        raise ValueError(f"Phase 1 Franka async server configuration mismatch: {mismatches}")
    if cfg.pretrained_name_or_path is None:
        raise ValueError("Phase 1 Franka async server requires --pretrained_name_or_path")
    if cfg.policy_device is None or not cfg.policy_device.startswith("cuda"):
        raise ValueError("Phase 1 Franka async server requires an explicit CUDA --policy_device")


@draccus.wrap()
def serve(cfg: PolicyServerConfig) -> None:
    """Start the stock LeRobot gRPC service with the Franka adapter."""

    validate_phase1_config(cfg)
    logging.info(pformat(asdict(cfg)))
    policy_server = FrankaPI0PolicyServer(cfg)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound_port = grpc_server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Franka PI0 policy server to {cfg.host}:{cfg.port}")

    policy_server.logger.info("Franka PI0 PolicyServer started on %s:%s", cfg.host, bound_port)
    grpc_server.start()
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("Keyboard interrupt received; stopping Franka PI0 PolicyServer")
    finally:
        policy_server.stop()
        grpc_server.stop(grace=1.0)


if __name__ == "__main__":
    serve()
