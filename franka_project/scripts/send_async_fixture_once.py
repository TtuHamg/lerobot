#!/usr/bin/env python
"""Send one frozen Franka observation and save one server response.

This is a non-actuating client: it always constructs the Franka plugin with
``dry_run=True``, sends exactly one observation, receives one action chunk,
writes that chunk to JSON, acknowledges delivery, and exits.  No returned
action is placed on a robot queue or forwarded to ROS/controller hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle  # nosec: this client is only for a trusted project policy server
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for source_root in (
    REPOSITORY_ROOT / "src",
    PROJECT_ROOT / "ros_lerobot/src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

DEFAULT_FIXTURE = PROJECT_ROOT / "fixtures/async/fastwam_grab_cups_observation_v1.npz"
DEFAULT_OUTPUT = Path("/tmp/fastwam_fixture_response.json")
DEFAULT_ACTION_LOG = Path("/tmp/fastwam_fixture_unexecuted_actions.jsonl")
DEFAULT_CALIBRATION_DIR = Path("/tmp/franka_fastwam_fixture_calibration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_fixture_ledger(fixture_path: Path, *, task: str) -> dict[str, Any]:
    ledger_path = fixture_path.with_suffix(".json")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"FastWAM fixture ledger does not exist: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise TypeError(f"FastWAM fixture ledger must be a JSON object: {ledger_path}")
    expected = {
        "fixture_kind": "fastwam_training_aligned_raw_observation",
        "policy_type": "fastwam",
        "fixture_sha256": _sha256(fixture_path),
    }
    drift = {
        key: {"expected": value, "actual": ledger.get(key)}
        for key, value in expected.items()
        if ledger.get(key) != value
    }
    wire = ledger.get("wire_contract")
    if not isinstance(wire, dict):
        drift["wire_contract"] = {"expected": "object", "actual": wire}
    else:
        expected_wire = {"fps": 30, "actions_per_chunk": 32, "rename_map": {}}
        for key, value in expected_wire.items():
            if wire.get(key) != value:
                drift[f"wire_contract.{key}"] = {"expected": value, "actual": wire.get(key)}
    source = ledger.get("source")
    source_task = source.get("task") if isinstance(source, dict) else None
    if source_task != task:
        drift["source.task"] = {"expected": task, "actual": source_task}
    if drift:
        raise ValueError(f"FastWAM fixture ledger contract drifted: {drift}")
    return ledger


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        path.chmod(0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def send_once(
    *,
    server_address: str,
    fixture_path: Path,
    output_path: Path,
    action_log_path: Path,
    calibration_dir: Path,
    task: str,
    timeout_s: float,
) -> dict[str, Any]:
    from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig
    from lerobot_robot_franka_ros.contract import validate_absolute_action

    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.helpers import TimedAction, TimedObservation
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.transport import services_pb2

    fixture_path = fixture_path.expanduser().resolve()
    if not fixture_path.is_file():
        raise FileNotFoundError(f"FastWAM fixture does not exist: {fixture_path}")
    fixture_ledger = _verify_fixture_ledger(fixture_path, task=task)
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be greater than zero")

    client = RobotClient(
        RobotClientConfig(
            robot=FrankaRosConfig(
                id="franka_fastwam_fixture_once",
                calibration_dir=calibration_dir.expanduser().resolve(),
                dry_run=True,
                fixture_path=fixture_path,
                action_log_path=action_log_path.expanduser().resolve(),
            ),
            server_address=server_address,
            task=task,
            policy_type="fastwam",
            pretrained_name_or_path="server-owned",
            policy_device="cpu",
            client_device="cpu",
            actions_per_chunk=32,
            action_offset=1,
            fps=30,
            chunk_size_threshold=0.0,
            aggregate_fn_name="latest_only",
            enable_pending_observation=True,
            pending_observation_timeout_s=timeout_s,
            rename_map={},
        )
    )
    try:
        if not client.start():
            raise RuntimeError("FastWAM policy handshake failed; inspect client/server logs")
        observation = client.robot.get_observation()
        observation["task"] = task
        observation_timestep = client._next_observation_timestep(client.latest_action)
        request = TimedObservation(
            timestamp=time.time(),
            timestep=observation_timestep,
            observation=observation,
            must_go=True,
        )
        if not client.send_observation(request):
            raise RuntimeError("the single fixture observation was not accepted by SendObservations")

        response = client.stub.GetActions(services_pb2.Empty(), timeout=timeout_s)
        if not response.data:
            raise RuntimeError("server returned an empty action delivery")
        timed_actions = pickle.loads(response.data)  # nosec: trusted project policy server
        if not isinstance(timed_actions, list) or len(timed_actions) != 32:
            raise ValueError(
                f"FastWAM response must contain 32 TimedAction values, got {type(timed_actions)} "
                f"length={len(timed_actions) if isinstance(timed_actions, list) else 'n/a'}"
            )
        if not all(isinstance(item, TimedAction) for item in timed_actions):
            raise TypeError("FastWAM response contains a value that is not TimedAction")
        if int(response.source_timestep) != observation_timestep:
            raise ValueError(
                "action delivery source timestep disagrees with the fixture request: "
                f"request={observation_timestep}, response={response.source_timestep}"
            )
        if not response.request_id or not response.chunk_id:
            raise ValueError("FastWAM response is missing the required request_id/chunk_id ACK identity")

        action_records = []
        for timed_action in timed_actions:
            action = timed_action.get_action().detach().cpu()
            if tuple(action.shape) != (8,):
                raise ValueError(f"absolute Franka action must have shape (8,), got {tuple(action.shape)}")
            action_dict = client._action_tensor_to_action_dict(action)
            action_records.append(
                {
                    "timestep": int(timed_action.get_timestep()),
                    "timestamp": float(timed_action.get_timestamp()),
                    "action": validate_absolute_action(
                        action_dict,
                        quaternion_norm_tolerance=client.robot.config.quaternion_norm_tolerance,
                    ),
                }
            )

        result = {
            "schema_version": 1,
            "policy_type": "fastwam",
            "fixture": str(fixture_path),
            "fixture_sha256": fixture_ledger["fixture_sha256"],
            "server_address": server_address,
            "task": task,
            "observation_timestep": observation_timestep,
            "request_id": str(response.request_id),
            "chunk_id": str(response.chunk_id),
            "source_timestep": int(response.source_timestep),
            "actions_executed": False,
            "actions": action_records,
        }
        _atomic_write_json(output_path, result)

        # Persisting the response JSON is this utility's local commit boundary.
        # ACK only after that succeeds, so the server can replay an interrupted delivery.
        client._remember_committed_action_chunk(str(response.chunk_id))
        client._resolve_pending_observation(str(response.request_id))
        client._ack_action_delivery(response)
        return result
    finally:
        client.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-log", type=Path, default=DEFAULT_ACTION_LOG)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--task", default="grab the paper cup.")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = send_once(
        server_address=args.server_address,
        fixture_path=args.fixture,
        output_path=args.output,
        action_log_path=args.action_log,
        calibration_dir=args.calibration_dir,
        task=args.task,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
