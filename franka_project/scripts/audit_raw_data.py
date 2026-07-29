#!/usr/bin/env python3
"""Run phase-D0 raw data audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from franka_eef_pipeline.audit import run_raw_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    report = run_raw_audit(args.config, max_episodes=args.max_episodes)
    summary = {
        "phase_status": report["phase_status"],
        "manifest_episodes": report["aggregate"]["manifest_episode_count"],
        "episodes": report["aggregate"]["episode_count"],
        "missing_raw_episodes": report["aggregate"]["missing_raw_episode_count"],
        "common_camera_anchors": report["aggregate"]["common_camera_anchors"],
        "valid_action15_anchors": report["aggregate"]["valid_action15_anchors"],
        "valid_action30_anchors": report["aggregate"]["valid_action30_anchors"],
        "output_json": report["output_json"],
        "output_markdown": report["output_markdown"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
