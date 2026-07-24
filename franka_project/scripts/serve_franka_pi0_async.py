#!/usr/bin/env python
"""Serve a project Franka PI0 or FastWAM checkpoint through async inference.

The historical filename is retained so existing deployment commands keep
working.  ``--policy_type`` selects the concrete servicer at startup.
"""

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

from franka_eef_pipeline.async_server import (  # noqa: E402
    FrankaServingContract,
    create_franka_policy_server,
    validate_franka_server_config,
)

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.transport import services_pb2_grpc  # noqa: E402


def validate_phase1_config(cfg: PolicyServerConfig) -> FrankaServingContract:
    """Fail fast when CLI inputs disagree with the selected checkpoint contract."""

    return validate_franka_server_config(cfg)


@draccus.wrap()
def serve(cfg: PolicyServerConfig) -> None:
    """Start the stock LeRobot gRPC service with the Franka adapter."""

    contract = validate_phase1_config(cfg)
    logging.info("Server configuration:\n%s", pformat(asdict(cfg)))
    logging.info("Checkpoint serving contract:\n%s", pformat(asdict(contract)))
    policy_server = create_franka_policy_server(cfg)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound_port = grpc_server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    if bound_port == 0:
        raise RuntimeError(
            f"Failed to bind Franka {cfg.policy_type} policy server to {cfg.host}:{cfg.port}"
        )

    policy_server.logger.info(
        "Franka %s PolicyServer started on %s:%s", cfg.policy_type, cfg.host, bound_port
    )
    grpc_server.start()
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info(
            "Keyboard interrupt received; stopping Franka %s PolicyServer", cfg.policy_type
        )
    finally:
        policy_server.stop()
        grpc_server.stop(grace=1.0)


if __name__ == "__main__":
    serve()
