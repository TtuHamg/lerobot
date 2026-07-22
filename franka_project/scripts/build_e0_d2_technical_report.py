#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the approved E0-D2 batch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "artifacts/raw_audit/d0_raw_audit.json"
SPIKE_PATH = PROJECT_ROOT / "artifacts/conversion/d2_cartesian_pipeline_spike.json"
OUTPUT_PATH = PROJECT_ROOT / "reports/e0_d2_report_artifact.json"


def _markdown(block_id: str, body: str, source_id: str | None = None) -> dict:
    block = {"id": block_id, "type": "markdown", "body": body.strip()}
    if source_id is not None:
        block["sourceId"] = source_id
    return block


def _table_block(block_id: str, table_id: str) -> dict:
    return {"id": block_id, "type": "table", "tableId": table_id, "layout": "full"}


def build_artifact() -> dict:
    audit = json.loads(AUDIT_PATH.read_text())
    spike = json.loads(SPIKE_PATH.read_text())
    manifest_path = Path(audit["manifest_path"])
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest_by_id = {item["episode_id"]: item for item in manifest["episodes"]}
    generated_at = datetime.now(timezone.utc).isoformat()

    missing_rows = []
    access_issues = []
    for missing in audit["missing_raw_episodes"]:
        episode_id = missing["episode_id"]
        meta = manifest_by_id[episode_id]
        missing_rows.append(
            {
                "episode_id": episode_id,
                "duration_s": float(meta["duration_s"]),
                "cam1_frames": int(meta["topic_counts"]["/camera1/camera1/color/image_raw"]),
                "raw_status": "MCAP missing; viz cache only",
            }
        )
        access_issues.append(
            {
                "id": f"missing_{episode_id}",
                "dataset": "raw_mcap",
                "message": f"{episode_id} raw MCAP is missing.",
            }
        )

    aggregate = audit["aggregate"]
    nominal_full_horizon = int(aggregate["common_camera_anchors"] - 50 * aggregate["episode_count"])
    coverage_rows = [
        {"order": 0, "metric": "Manifest PASS episodes", "value": 25, "unit": "episodes", "interpretation": "intended raw scope"},
        {"order": 1, "metric": "Readable raw episodes", "value": 22, "unit": "episodes", "interpretation": "88% availability"},
        {"order": 2, "metric": "Missing raw episodes", "value": 3, "unit": "episodes", "interpretation": "12% missing; Critical blocker"},
        {"order": 3, "metric": "Raw cam1 frames (readable)", "value": aggregate["raw_cam1_frames"], "unit": "frames", "interpretation": "22 readable episodes"},
        {"order": 4, "metric": "Common-interval cam1 anchors", "value": aggregate["common_camera_anchors"], "unit": "anchors", "interpretation": "98.086% of readable raw cam1"},
        {"order": 5, "metric": "Nominal full-horizon anchors", "value": nominal_full_horizon, "unit": "anchors", "interpretation": "episode-tail gate only"},
        {"order": 6, "metric": "Valid full-horizon anchors", "value": aggregate["valid_action15_anchors"], "unit": "anchors", "interpretation": "after source-age gate"},
    ]

    age_rows = []
    gates = {"cam2": 100.0, "eef": 50.0, "gripper": 10.0, "qpos_audit": 10.0}
    labels = {"cam2": "cam2", "eef": "EEF", "gripper": "gripper", "qpos_audit": "qpos audit"}
    for order, name in enumerate(("cam2", "eef", "gripper", "qpos_audit")):
        item = aggregate["alignment_age_ms"][name]
        age_rows.append(
            {
                "order": order,
                "stream": labels[name],
                "p50_ms": item["p50"],
                "p95_ms": item["p95"],
                "p99_ms": item["p99"],
                "max_ms": item["max"],
                "gate_ms": gates[name],
            }
        )

    d2_rows = []
    for profile_name, display_name in (("action15", "obs15/action15"), ("action30", "obs15/action30")):
        item = spike["profiles"][profile_name]
        d2_rows.append(
            {
                "profile": display_name,
                "action_fps": item["action_fps"],
                "chunk_size": item["chunk_size"],
                "shape": "x".join(str(value) for value in item["action_chunk_shape"]),
                "anchors": item["anchor_count"],
                "rotation_error_rad": item["geometry_roundtrip"]["rotation_max_geodesic_error_rad"],
                "max_theta_rad": item["geometry_roundtrip"]["so3_theta_max_rad"],
                "action_norm_error": item["normalization_roundtrip"]["action_max_abs_error"],
            }
        )

    sources = [
        {"id": "e0_environment", "label": "E0 environment manifest", "path": "environment/e0_environment_manifest.json"},
        {"id": "d0_audit", "label": "D0 raw audit", "path": "artifacts/raw_audit/d0_raw_audit.json"},
        {"id": "d1_contract", "label": "D1 frozen data contract", "path": "configs/data/franka_current_eef_contract_v1.yaml"},
        {"id": "d2_spike", "label": "D2 Cartesian spike", "path": "artifacts/conversion/d2_cartesian_pipeline_spike.json"},
    ]

    tables = [
        {
            "id": "phase_table",
            "title": "E0-D3 phase status",
            "subtitle": "The approved batch ends at D2; D3 has not started.",
            "dataset": "phase_status",
            "sourceId": "d1_contract",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "phase", "label": "Phase", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "next_gate", "label": "Next gate", "type": "text"},
            ],
        },
        {
            "id": "coverage_table",
            "title": "Raw completeness and usable samples",
            "subtitle": "Manifest grain is episode; alignment grain is cam1 observation anchor.",
            "dataset": "coverage",
            "sourceId": "d0_audit",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "value", "label": "Value", "format": "number"},
                {"field": "unit", "label": "Unit", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "missing_table",
            "title": "Missing PASS episodes",
            "subtitle": "The three entries total 90.361 s and 1,321 manifest cam1 frames.",
            "dataset": "missing_episodes",
            "sourceId": "d0_audit",
            "defaultSort": {"field": "episode_id", "direction": "asc"},
            "columns": [
                {"field": "episode_id", "label": "Episode", "type": "text"},
                {"field": "duration_s", "label": "Manifest duration (s)", "format": "number"},
                {"field": "cam1_frames", "label": "Manifest cam1 frames", "format": "number"},
                {"field": "raw_status", "label": "Raw status", "type": "text"},
            ],
        },
        {
            "id": "alignment_table",
            "title": "Causal alignment source age",
            "subtitle": "22 readable episodes; cam1 log time is the observation clock; values in ms.",
            "dataset": "alignment_age",
            "sourceId": "d0_audit",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "stream", "label": "Stream", "type": "text"},
                {"field": "p50_ms", "label": "p50 (ms)", "format": "number"},
                {"field": "p95_ms", "label": "p95 (ms)", "format": "number"},
                {"field": "p99_ms", "label": "p99 (ms)", "format": "number"},
                {"field": "max_ms", "label": "max (ms)", "format": "number"},
                {"field": "gate_ms", "label": "Gate (ms)", "format": "number"},
            ],
        },
        {
            "id": "d2_table",
            "title": "D2 dual-rate Cartesian numerical acceptance",
            "subtitle": "Two episodes and 824 complete 50-camera-interval horizon anchors.",
            "dataset": "d2_profiles",
            "sourceId": "d2_spike",
            "defaultSort": {"field": "action_fps", "direction": "asc"},
            "columns": [
                {"field": "profile", "label": "Profile", "type": "text"},
                {"field": "action_fps", "label": "Action Hz", "format": "number"},
                {"field": "chunk_size", "label": "K", "format": "number"},
                {"field": "shape", "label": "Action shape", "type": "text"},
                {"field": "anchors", "label": "Anchors", "format": "number"},
                {"field": "rotation_error_rad", "label": "Rotation RT (rad)", "format": "number"},
                {"field": "max_theta_rad", "label": "Max theta (rad)", "format": "number"},
                {"field": "action_norm_error", "label": "Norm RT", "format": "number"},
            ],
        },
    ]

    blocks = [
        _markdown("title", "# Frank3 E0-D2 data and pipeline validation"),
        _markdown(
            "technical_summary",
            """
## Technical summary

- **E0 passed:** Conda, the provided loader, two-GPU BF16/cuDNN/NCCL, local pi0_base, and W&B scalar logging all work. W&B logged zero artifacts.
- **D0 is blocked:** only 22 of 25 PASS episodes have raw MCAP files, so a reproducible 25-episode conversion cannot start.
- **D1 conditionally passed:** 10D current-EEF state, 7D anchor-relative Cartesian action, common-interval alignment, gripper mapping, and both rate profiles are frozen. No A/B or held-out val/test split is used.
- **D2 passed:** causal time selection, endpoint identity, SO(3) encode/decode, and normalization round-trips passed on 824 complete horizons.

**Decision: stop at D2.** Restore the three MCAP files, or explicitly approve a versioned 22of25 scope, before D3.
            """,
        ),
        _markdown(
            "scope_definitions",
            """
## Scope and denominators

Raw completeness uses the 25 PASS episodes in the manifest. Timestamp and schema results use the 22 currently readable episodes. D2 is a numerical spike on two named episodes; it performs no image conversion, no full LeRobot conversion, and no pi0 training.

A valid full-horizon anchor has all 50 future camera intervals and passes every frozen source-age gate. Episode-tail padding is never treated as real action data.
            """,
        ),
        _table_block("phase_status_block", "phase_table"),
        _markdown(
            "raw_finding",
            """
## Three missing MCAP files are the only Critical completeness failure

Across the 22 readable episodes, topic counts match the manifest, images are consistently 480x640 RGB8, and EEF/gripper/qpos contain no non-finite values or schema drift. The three missing PASS entries have only visualization caches, which cannot reproduce the original observations and timestamps.
            """,
            "d0_audit",
        ),
        _table_block("coverage_block", "coverage_table"),
        _table_block("missing_block", "missing_table"),
        _markdown(
            "alignment_finding",
            f"""
## The frozen cam2 age gate retains 96.76% of nominal complete horizons

The common interval contains {aggregate['common_camera_anchors']:,} cam1 anchors. Episode-tail constraints alone yield {nominal_full_horizon:,} complete horizons; applying the 100/50/10 ms cam2/EEF/gripper gates retains {aggregate['valid_action15_anchors']:,}, a reduction of 250 (3.24%). All 18 invalid observations are caused by local cam2 staleness; EEF and gripper max ages are 33.70 ms and 3.92 ms.
            """,
            "d0_audit",
        ),
        _table_block("alignment_block", "alignment_table"),
        _markdown(
            "d2_finding",
            """
## Both action-rate profiles pass every numerical gate

The 15 Hz profile uses t+1 through t+50. The 30 Hz profile uses midpoint then endpoint for each camera interval, producing 100 slots. Valid-anchor sets match, and every 30 Hz endpoint value and source timestamp is bit-exact with its 15 Hz counterpart.

Maximum position/rotation/gripper round-trip errors are 0 m, 5.162e-8 rad, and 0. Maximum relative rotation is 0.981708 rad, safely below pi-0.1. Normalize then unnormalize has maximum error 1.110e-16.
            """,
            "d2_spike",
        ),
        _table_block("d2_block", "d2_table"),
        _markdown(
            "methodology",
            """
## Frozen method and feature contract

- Observation clock: cam1 MCAP log_time_ns with causal latest-not-after alignment.
- State: 10D xyz + rotation6D + gripper; qpos remains audit-only.
- Action: 7D base-frame delta xyz + body-frame SO(3) rotvec + target gripper, anchored at current measured EEF.
- Targets: future measured EEF/gripper, explicitly a realized-waypoint proxy rather than an unrecorded desired command.
- Statistics: min/max/mean/std/count/q01/q10/q50/q90/q99 on the effective 10D state and 7D action; 15 Hz and 30 Hz statistics remain separate.
- Split: no A/B or held-out validation/test episodes; a future train-anchor subset may monitor in-distribution val loss only.
            """,
        ),
        _markdown(
            "limitations",
            """
## Limitations and robustness boundary

1. D2 does not clear the D0 missing-data blocker and does not cover the unavailable episodes.
2. current_pose is the measured configured EE frame in base, but the bag does not contain numerical F_T_EE/NE_T_EE. Real-robot deployment must reuse the same configured EE or apply an explicit fixed transform.
3. There is no held-out evaluation. D2 proves numerical geometry and alignment behavior, not task success or generalization.
4. Version 1 records chase-frame/irregular timing evidence but does not rewrite real time; stale cam2 observations are excluded only through the explicit age gate.
            """,
        ),
        _markdown(
            "next_steps",
            """
## Recommended next steps

1. Prefer restoring the three missing MCAP files, then rerun D0 to reach 25/25.
2. If restoration is impossible and rapid verification is acceptable, explicitly approve a new 22of25 derived manifest and dataset version.
3. After D3 is unblocked, convert both profiles and compute their full statistics; train first on obs15/action15 only after D4, S0, and S1 pass.
4. Confirm that collection and deployment use the same configured EE/TCP before a real-robot rollout.
            """,
        ),
        _markdown(
            "further_questions",
            """
## Decision required

Should the project restore the three raw MCAP files and retain the 25-episode scope, or should it create an explicitly approved, versioned 22of25 rapid-verification scope?
            """,
        ),
    ]

    phase_rows = [
        {"order": 0, "phase": "E0", "status": "PASS", "evidence": "Conda/loader/BF16/cuDNN/NCCL/pi0_base/W&B scalar", "next_gate": "complete"},
        {"order": 1, "phase": "D0", "status": "BLOCKED", "evidence": "22/25 raw MCAP readable", "next_gate": "restore 3 or authorize 22of25"},
        {"order": 2, "phase": "D1", "status": "CONDITIONAL PASS", "evidence": "EEF/time/gripper/dual-rate contract frozen", "next_gate": "usable for D2, not D3"},
        {"order": 3, "phase": "D2", "status": "PASS", "evidence": "824 anchors; all numerical checks passed", "next_gate": "review"},
        {"order": 4, "phase": "D3", "status": "NOT STARTED", "evidence": "outside approved batch and D0 blocked", "next_gate": "explicitly unblock scope"},
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Frank3 E0-D2 data and pipeline validation",
            "description": "Technical gate report for Frank3 raw integrity, EEF contract, and dual-rate Cartesian action construction.",
            "generatedAt": generated_at,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "datasets": {
                "phase_status": phase_rows,
                "coverage": coverage_rows,
                "missing_episodes": missing_rows,
                "alignment_age": age_rows,
                "d2_profiles": d2_rows,
            },
            "accessIssues": access_issues,
        },
        "sources": sources,
    }


def main() -> None:
    artifact = build_artifact()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "written", "output": str(OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
