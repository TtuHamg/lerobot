"""Non-actuating fixture source and JSONL action sink."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TextIO

from lerobot.types import RobotAction, RobotObservation

from .contract import DryRunObservation, load_dry_run_observation


class DryRunBackend:
    """A deliberately small backend that cannot communicate with hardware."""

    def __init__(self, *, fixture_path: Path | None, action_log_path: Path | None, robot_id: str | None):
        self.fixture_path = fixture_path
        self.action_log_path = action_log_path
        self.robot_id = robot_id
        self._fixture: DryRunObservation | None = None
        self._action_log: TextIO | None = None
        self._sequence = 0
        self._write_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._fixture is not None and self._action_log is not None and not self._action_log.closed

    def connect(self) -> None:
        if self.fixture_path is None:
            raise ValueError("fixture_path is required in dry-run mode")
        if self.action_log_path is None:
            raise ValueError("action_log_path is required in dry-run mode")
        if self.fixture_path.expanduser().resolve() == self.action_log_path.expanduser().resolve():
            raise ValueError("fixture_path and action_log_path must refer to different files")

        fixture = load_dry_run_observation(self.fixture_path)
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        action_log = self.action_log_path.open("a", encoding="utf-8")

        # Publish connection state only after every resource is valid/open.
        self._fixture = fixture
        self._action_log = action_log
        self._sequence = 0

    def get_observation(self) -> RobotObservation:
        if not self.is_connected or self._fixture is None:
            raise RuntimeError("Dry-run backend is not connected")
        return self._fixture.as_robot_observation()

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected or self._action_log is None:
            raise RuntimeError("Dry-run backend is not connected")

        with self._write_lock:
            record = {
                "schema_version": 1,
                "dry_run": True,
                "robot_id": self.robot_id,
                "sequence": self._sequence,
                "monotonic_ns": time.monotonic_ns(),
                "action": action,
            }
            payload = json.dumps(record, allow_nan=False, separators=(",", ":"))
            self._action_log.write(payload + "\n")
            self._action_log.flush()
            self._sequence += 1
        return action

    def disconnect(self) -> None:
        action_log = self._action_log
        self._action_log = None
        self._fixture = None
        if action_log is not None:
            action_log.flush()
            action_log.close()
