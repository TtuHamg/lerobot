"""Debug pi0 LoRA fine-tuning through the normal lerobot-train entrypoint.

Run from the repository root:

    PYTHONPATH=src python tools/debug_pi0_lora_train.py

Then attach VS Code/debugpy to 127.0.0.1:4567.

This script intentionally uses a tiny debug run by default:
    - steps=2
    - batch_size=1
    - wandb disabled
    - checkpoint saving disabled
    - torch compile disabled

It still goes through `lerobot.scripts.lerobot_train.main()`, so breakpoints in
dataset loading, pi0 construction, PEFT wrapping, and the training step behave
like a regular `lerobot-train ...` invocation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/m2v_intern/tujiahang/Projects/lerobot")
BASE_PI0 = Path(
    "/ytech_milm_intern/tujiahang/.cache/huggingface/hub/"
    "models--lerobot--pi0_base/snapshots/25c379b52ba2ff8788cab921758a3cc3fe3f77f2"
)
DATASET_ROOT = Path(
    "/ytech_milm_intern/tujiahang/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_task0_new"
)


def maybe_wait_for_debugger() -> None:
    """Open a debugpy attach port unless disabled with WAIT_FOR_DEBUGPY=0."""
    if os.environ.get("WAIT_FOR_DEBUGPY", "1") == "0":
        return

    import debugpy

    host = os.environ.get("DEBUGPY_HOST", "127.0.0.1")
    port = int(os.environ.get("DEBUGPY_PORT", "4567"))
    debugpy.listen((host, port))
    print(f"Waiting for debugger attach on {host}:{port} ...")
    debugpy.wait_for_client()
    print("Debugger attached.")


def build_lerobot_train_argv() -> list[str]:
    """Build CLI args exactly as if calling `lerobot-train ...`."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "output_lerobot" / "pi0" / "debug_task0_lora" / timestamp

    return [
        "lerobot-train",
        f"--policy.path={BASE_PI0}",
        "--policy.push_to_hub=false",
        "--policy.empty_cameras=0",
        "--rename_map={"
        '"observation.images.front":"observation.images.base_0_rgb",'
        '"observation.images.wrist.left":"observation.images.left_wrist_0_rgb"'
        "}",
        "--dataset.repo_id=tuuy/lerobot_so_arm101_task0_new",
        f"--dataset.root={DATASET_ROOT}",
        "--dataset.streaming=false",
        f"--output_dir={output_dir}",
        "--job_name=debug_pi0_task0_lora",
        "--peft.method_type=LORA",
        "--peft.r=16",
        "--peft.lora_alpha=32",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--policy.dtype=bfloat16",
        "--steps=2",
        "--save_checkpoint=false",
        "--save_freq=1",
        "--eval_freq=1000000",
        "--log_freq=1",
        "--batch_size=1",
        "--num_workers=0",
        "--persistent_workers=false",
        "--wandb.enable=false",
        "--wandb.disable_artifact=true",
    ]


def main() -> None:
    os.chdir(REPO_ROOT)

    # Keep this optional so you can choose the GPU externally:
    #   CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python tools/debug_pi0_lora_train.py
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # maybe_wait_for_debugger()

    sys.argv = build_lerobot_train_argv()
    print("Running equivalent command:")
    print(" ".join(sys.argv))

    # Stop here if you want to inspect or edit sys.argv before draccus parses it.
    breakpoint()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
