"""No-ROS contract tests executing real client methods without optional device imports."""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
ASYNC_ROOT = REPO_ROOT / "src/lerobot/async_inference"
CLIENT_PATH = (
    REPO_ROOT / "franka_project/ros_lerobot/src/lerobot_robot_franka_ros/joint_ros2_client.py"
)
SCRIPT_PATH = REPO_ROOT / "franka_project/scripts/run_joint_client.sh"


def _load_definitions(path, definitions, namespace):
    parsed = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in parsed.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)) or node.name not in definitions:
            continue
        methods = definitions[node.name]
        if methods is not None:
            node.body = [
                method for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in methods
            ]
        selected.append(node)
    assert len(selected) == len(definitions)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _client_types():
    namespace = {
        "__name__": __name__, "dataclass": dataclass, "field": field,
        "Queue": Queue, "time": time, "DEFAULT_FPS": 30,
        "AGGREGATE_FUNCTIONS": {"latest_only": lambda old, new: new},
    }
    _load_definitions(
        ASYNC_ROOT / "helpers.py",
        {"TimedData": None, "TimedAction": None, "TimedObservation": None}, namespace,
    )
    _load_definitions(
        ASYNC_ROOT / "configs.py",
        {"RobotClientConfig": None, "get_aggregate_function": None}, namespace,
    )
    _load_definitions(
        ASYNC_ROOT / "robot_client.py",
        {"RobotClient": {
            "running", "send_observation", "_aggregate_action_queues", "_ready_to_send_observation",
            "control_loop_observation", "control_loop_action", "_resolve_pending_observation",
            "_claim_pending_observation_retry",
        }}, namespace,
    )
    _load_definitions(
        CLIENT_PATH,
        {"FrankaJointRos2ClientConfig": None, "FrankaJointRos2RobotClient": {
            "send_observation", "_aggregate_action_queues", "_ready_to_send_observation",
        }}, namespace,
    )
    return SimpleNamespace(**namespace)


class ReplanContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types = _client_types()

    def config(self, **overrides):
        values = dict(
            robot=SimpleNamespace(), actions_per_chunk=32, replan_steps=8,
            chunk_size_threshold=0.0, enable_pending_observation=True,
            aggregate_fn_name="latest_only", pending_observation_timeout_s=15, fps=30,
        )
        values.update(overrides)
        return self.types.FrankaJointRos2ClientConfig(**values)

    def client(self, steps=8):
        client = object.__new__(self.types.FrankaJointRos2RobotClient)
        client.config = self.config(replan_steps=steps)
        client.latest_action_lock = threading.Lock()
        client.latest_action = -1
        client.action_queue_lock = threading.Lock()
        client.action_queue = Queue()
        client.action_queue_size = []
        client.action_chunk_size = 32
        client._chunk_size_threshold = 0.0
        client._pending_observation_lock = threading.Lock()
        client._pending_observation = False
        client._pending_observation_sent_at = None
        client._pending_observation_request_id = None
        client._pending_observation_value = None
        client._pending_observation_retry_due = False
        client._next_observation_sequence = 0
        client._client_session_id = "offline-replan-test"
        client.shutdown_event = threading.Event()
        client.must_go = threading.Event()
        client.logger = logging.getLogger("offline-replan-test")
        client.robot = SimpleNamespace(
            published=[], frame=0, gateway_ready=True,
            get_observation=lambda: {"frame": client.robot.frame},
            send_action=lambda action: action,
        )
        client.robot._backend = SimpleNamespace(
            ready_for_next_observation=lambda: client.robot.gateway_ready,
        )
        client.robot.publish_action_chunk = lambda actions, **metadata: client.robot.published.append(
            (actions, metadata)
        )
        client.observations = []
        client._send_observation_rpc = lambda obs: client.observations.append(obs) or True
        client._action_tensor_to_action_dict = lambda action: {"target": action}
        return client

    def response(self, observation, count=32):
        return [self.types.TimedAction(
            timestamp=observation.timestamp + index / 30,
            timestep=observation.timestep + index,
            action=observation.observation["frame"] + index + 1,
        ) for index in range(count)]

    def test_config_validation_and_serialization(self):
        for value in [1, 8, 16, 32, None]:
            with self.subTest(value=value):
                config = self.config(replan_steps=value)
                self.assertEqual(config.to_dict()["replan_steps"], value)
        for value in [0, -1, 33, True, 1.5, "8"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.config(replan_steps=value)
        for overrides in [
            {"chunk_size_threshold": 0.5}, {"enable_pending_observation": False},
            {"aggregate_fn_name": "weighted_average"}, {"actions_per_chunk": 4},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)

    def test_repeated_replans_keep_exact_prefix_and_first_target(self):
        for steps in [1, 8, 16, 32]:
            with self.subTest(steps=steps):
                client = self.client(steps)
                for cycle in range(4):
                    self.assertTrue(client._ready_to_send_observation())
                    client.control_loop_observation("stack yellow block")
                    observation = client.observations[-1]
                    self.assertEqual(observation.timestep, cycle * steps)
                    self.assertFalse(client._ready_to_send_observation())
                    response = self.response(observation)
                    client._aggregate_action_queues(response)
                    client._resolve_pending_observation(observation.request_id)
                    published, metadata = client.robot.published[-1]
                    self.assertEqual(published, response[:steps])
                    self.assertEqual(list(client.action_queue.queue), response[:steps])
                    self.assertEqual(len(response), 32)
                    self.assertEqual(metadata["source_observation_timestep"], observation.timestep)
                    self.assertEqual(metadata["source_observation_timestamp"], observation.timestamp)
                    self.assertEqual(metadata["period_s"], 1 / 30)
                    self.assertEqual(published[0].action, cycle * steps + 1)
                    client.robot.gateway_ready = False
                    while not client.action_queue.empty():
                        self.assertFalse(client._ready_to_send_observation())
                        client.control_loop_action()
                    self.assertFalse(client._ready_to_send_observation())
                    client.robot.frame = published[-1].action
                    client.robot.gateway_ready = True
                    self.assertTrue(client._ready_to_send_observation())

    def test_discarded_suffix_is_not_promoted_by_stale_filter(self):
        client = self.client()
        client.control_loop_observation("stack yellow block")
        response = self.response(client.observations[-1])
        client.latest_action = 3
        client._aggregate_action_queues(response)
        self.assertEqual(client.robot.published[-1][0], response[4:8])
        client.latest_action = 7
        client.action_queue = Queue()
        client._aggregate_action_queues(response)
        self.assertEqual(len(client.robot.published), 1)
        self.assertTrue(client.action_queue.empty())

    def test_short_response_empty_response_and_bad_types(self):
        client = self.client()
        client.control_loop_observation("stack yellow block")
        response = self.response(client.observations[-1], count=3)
        client._aggregate_action_queues(response)
        self.assertEqual(client.robot.published[-1][0], response)
        client._aggregate_action_queues([])
        self.assertEqual(list(client.action_queue.queue), response)
        for invalid in [(), [object()]]:
            with self.assertRaises(TypeError):
                client._aggregate_action_queues(invalid)

    def test_transport_retry_preserves_tick_timestamp_and_request(self):
        client = self.client()
        client.control_loop_observation("stack yellow block")
        observation = client.observations[-1]
        identity = (observation.timestep, observation.timestamp, observation.request_id)
        client.latest_action = 7
        client._pending_observation_retry_due = True
        client.control_loop_observation("stack yellow block")
        self.assertIs(client.observations[-1], observation)
        self.assertEqual((observation.timestep, observation.timestamp, observation.request_id), identity)
        client.send_observation(observation)
        self.assertEqual((observation.timestep, observation.timestamp, observation.request_id), identity)

    def test_no_replan_preserves_previous_tick_and_full_chunk_behavior(self):
        client = self.client(None)
        for expected_count in [32, 31]:
            client.control_loop_observation("stack yellow block")
            observation = client.observations[-1]
            self.assertEqual(observation.timestep, max(client.latest_action, 0))
            client._aggregate_action_queues(self.response(observation))
            self.assertEqual(len(client.robot.published[-1][0]), expected_count)
            while not client.action_queue.empty():
                client.control_loop_action()

    def test_publication_failure_stops_client(self):
        client = self.client()
        client.control_loop_observation("stack yellow block")

        def reject_publication(*args, **kwargs):
            raise RuntimeError("test gateway failure")

        client.robot.publish_action_chunk = reject_publication
        with self.assertLogs(client.logger, level="ERROR"), self.assertRaises(RuntimeError):
            client._aggregate_action_queues(self.response(client.observations[-1]))
        self.assertTrue(client.shutdown_event.is_set())


class ReplanLauncherTests(unittest.TestCase):
    def run_argument_parser(self, arguments, environment=None):
        source = SCRIPT_PATH.read_text()
        parser = source.split('SERVER_ADDRESS="', 1)[0]
        launcher = source.rsplit('cd "$REPO_ROOT"', 1)[1]
        with tempfile.TemporaryDirectory() as directory:
            harness = parser + '''
ROBOT_ARGS=(--robot.action_space="$ACTION_SPACE")
SERVER_ADDRESS=127.0.0.1:15174
LOG_DIR="$HOME"
CLIENT_LOG_NAME=arguments.log
python() { printf '%s\\n' "$@"; }
''' + launcher
            env = {**os.environ, "HOME": directory}
            env.pop("REPLAN_STEPS", None)
            env.pop("ACTION_SPACE", None)
            env.update(environment or {})
            return subprocess.run(
                ["bash", "-c", harness, str(SCRIPT_PATH), *arguments],
                env=env, capture_output=True, text=True, timeout=10,
            )

    def test_argument_forwarding_and_environment_override(self):
        cases = [
            (["--qpos", "--replan", "8", "stack yellow block"], {}, "8", "qpos"),
            (["--eef", "--replan=16", "stack yellow block"], {}, "16", "eef"),
            (["stack yellow block"], {"REPLAN_STEPS": "4"}, "4", "qpos"),
            (["--replan", "8"], {"REPLAN_STEPS": "4"}, "8", "qpos"),
            ([], {}, None, "qpos"),
        ]
        for arguments, environment, expected, action_space in cases:
            with self.subTest(arguments=arguments, environment=environment):
                result = self.run_argument_parser(arguments, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = result.stdout.splitlines()
                self.assertIn("--actions_per_chunk=32", values)
                self.assertIn(f"--robot.action_space={action_space}", values)
                if expected is None:
                    self.assertFalse(any(value.startswith("--replan_steps=") for value in values))
                else:
                    self.assertIn(f"--replan_steps={expected}", values)
                if "stack yellow block" in arguments:
                    self.assertIn("--task=stack yellow block", values)

    def test_invalid_values_fail_before_ros_or_robot_setup(self):
        cases = [["--replan"], ["--replan="], ["--replan", ""], ["--replan", "--qpos"]]
        cases += [["--replan", value] for value in ["0", "-1", "33", "1.5", "eight", "08"]]
        for arguments in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                env = {**os.environ, "HOME": directory}
                env.pop("REPLAN_STEPS", None)
                result = subprocess.run(
                    ["bash", str(SCRIPT_PATH), *arguments], env=env,
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("ERROR: --replan", result.stdout)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_help_has_replan(self):
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "--help"], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--replan N", result.stdout)


if __name__ == "__main__":
    unittest.main()
