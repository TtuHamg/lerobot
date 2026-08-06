# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}

OBSERVATION_TRIGGER_MODES = ("queue_and_plan", "post_action_delay")


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )
    observation_similarity_mode: str = field(
        default="none",
        metadata={
            "help": "Observation similarity filtering mode. 'none' disables filtering; "
            "'state' skips observations whose state is similar to the last processed observation."
        },
    )

    # Policy configuration. When set, these values take precedence over policy
    # settings sent by the robot client.
    policy_type: str | None = field(default=None, metadata={"help": "Type of policy to serve"})
    pretrained_name_or_path: str | None = field(
        default=None, metadata={"help": "Pretrained model name or path"}
    )
    actions_per_chunk: int | None = field(
        default=None, metadata={"help": "Number of actions returned per chunk"}
    )
    policy_device: str | None = field(default=None, metadata={"help": "Device for policy inference"})
    fastwam_joint_video_inference: bool = field(
        default=False,
        metadata={
            "help": "For policy_type='fastwam', jointly denoise the checkpoint's future video and action. "
            "The generated video stays server-local; only actions are returned to the client."
        },
    )
    fastwam_joint_video_output_dir: str | None = field(
        default=None,
        metadata={
            "help": "Optional directory for MP4s from FastWAM joint video inference. "
            "Requires fastwam_joint_video_inference=true."
        },
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

        if self.observation_similarity_mode not in ("none", "state"):
            raise ValueError(
                "observation_similarity_mode must be one of ('none', 'state'), "
                f"got {self.observation_similarity_mode!r}"
            )

        if self.actions_per_chunk is not None and self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.policy_type == "":
            raise ValueError("policy_type cannot be empty")

        if self.pretrained_name_or_path == "":
            raise ValueError("pretrained_name_or_path cannot be empty")

        if self.policy_device == "":
            raise ValueError("policy_device cannot be empty")

        if self.fastwam_joint_video_output_dir == "":
            raise ValueError("fastwam_joint_video_output_dir cannot be empty")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
            "observation_similarity_mode": self.observation_similarity_mode,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "actions_per_chunk": self.actions_per_chunk,
            "policy_device": self.policy_device,
            "fastwam_joint_video_inference": self.fastwam_joint_video_inference,
            "fastwam_joint_video_output_dir": self.fastwam_joint_video_output_dir,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policy configuration. These are optional when configured on the policy server.
    policy_type: str | None = field(default=None, metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str | None = field(
        default=None, metadata={"help": "Pretrained model name or path"}
    )

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int | None = field(default=None, metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})
    interactive_task_control: bool = field(
        default=False,
        metadata={"help": "Select, stop, and reassign server-advertised tasks from the keyboard"},
    )
    gateway_arm_timeout_s: float = field(
        default=5.0,
        metadata={"help": "Timeout for the external Franka safety gateway set_armed service"},
    )

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(
        default="cpu",
        metadata={"help": "Device for policy inference; overridden by server configuration when set"},
    )
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    action_offset: int = field(
        default=0,
        metadata={
            "help": "Offset added to latest_action when assigning a new observation timestep. "
            "Set to 1 so the first predicted action targets the next timestep."
        },
    )
    enable_pending_observation: bool = field(
        default=True,
        metadata={
            "help": "If true, send at most one in-flight observation until an action chunk is received or "
            "pending_observation_timeout_s expires."
        },
    )
    pending_observation_timeout_s: float = field(
        default=2.0,
        metadata={
            "help": "Maximum time to wait for an action chunk after sending an observation before sending "
            "another observation."
        },
    )
    observation_trigger_mode: str = field(
        default="queue_and_plan",
        metadata={
            "help": "Observation trigger mode. 'queue_and_plan' preserves the normal queue threshold and "
            "robot execution gates. 'post_action_delay' ignores those gates and sends the next observation "
            "only after post_action_observation_delay_s has elapsed since a successfully committed action "
            "chunk."
        },
    )
    post_action_observation_delay_s: float = field(
        default=5.0,
        metadata={
            "help": "Delay from a successfully committed action chunk to the next observation when "
            "observation_trigger_mode='post_action_delay'."
        },
    )
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )
    rename_map: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Observation key mapping from robot keys to policy keys."},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if self.policy_type == "":
            raise ValueError("policy_type cannot be empty")

        if self.pretrained_name_or_path == "":
            raise ValueError("pretrained_name_or_path cannot be empty")

        if self.policy_device == "":
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if isinstance(self.action_offset, bool) or not isinstance(self.action_offset, int):
            raise ValueError(f"action_offset must be an integer, got {self.action_offset!r}")
        if self.action_offset not in (0, 1):
            raise ValueError(f"action_offset must be 0 or 1, got {self.action_offset}")

        if self.enable_pending_observation and self.pending_observation_timeout_s <= 0:
            raise ValueError(
                f"pending_observation_timeout_s must be positive, got {self.pending_observation_timeout_s}"
            )
        if self.gateway_arm_timeout_s <= 0:
            raise ValueError(
                f"gateway_arm_timeout_s must be positive, got {self.gateway_arm_timeout_s}"
            )

        if self.observation_trigger_mode not in OBSERVATION_TRIGGER_MODES:
            raise ValueError(
                f"observation_trigger_mode must be one of {OBSERVATION_TRIGGER_MODES}, "
                f"got {self.observation_trigger_mode!r}"
            )

        if (
            isinstance(self.post_action_observation_delay_s, bool)
            or not isinstance(self.post_action_observation_delay_s, (int, float))
            or not math.isfinite(float(self.post_action_observation_delay_s))
            or self.post_action_observation_delay_s <= 0
        ):
            raise ValueError(
                "post_action_observation_delay_s must be a finite positive number, "
                f"got {self.post_action_observation_delay_s!r}"
            )

        if self.observation_trigger_mode == "post_action_delay" and not self.enable_pending_observation:
            raise ValueError(
                "observation_trigger_mode='post_action_delay' requires "
                "enable_pending_observation=true to prevent repeated observations before the next action "
                "chunk arrives"
            )

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk is not None and self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "action_offset": self.action_offset,
            "enable_pending_observation": self.enable_pending_observation,
            "pending_observation_timeout_s": self.pending_observation_timeout_s,
            "observation_trigger_mode": self.observation_trigger_mode,
            "post_action_observation_delay_s": self.post_action_observation_delay_s,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "interactive_task_control": self.interactive_task_control,
            "gateway_arm_timeout_s": self.gateway_arm_timeout_s,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
            "rename_map": self.rename_map,
        }
