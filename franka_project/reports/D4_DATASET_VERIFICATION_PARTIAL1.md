# D4 Franka LeRobot dataset verification

- Status: **PASS_PARTIAL_SMOKE**
- Generated: `2026-07-15T12:48:55.556629+00:00`
- Config: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/configs/data/franka_current_eef_conversion_22of25_v1.yaml`
- Action-15 root: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/.cache/d3_smoke/franka_current_eef_obs15_act15_v1_22of25_76b839b2_partial1`
- Action-30 root: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/.cache/d3_smoke/franka_current_eef_obs15_act30_v1_22of25_76b839b2_partial1`

## Verified counts

```json
{
  "episodes": 1,
  "common_camera_anchors": 587,
  "main_rows": 586,
  "valid_anchors": 468,
  "action15_effective_stats_count": 23400,
  "action30_carrier_rows": 1172,
  "action30_effective_stats_count": 46800
}
```

## Video QA

- Random/boundary decoded samples: [0, 118, 305, 497, 585]
- Raw-MCAP comparisons: 6
- Raw-MCAP maximum MAE: 2.2364442274305554
- Raw-MCAP minimum PSNR (dB): 38.10214006370492
- H.264 source comparison is intentionally thresholded, not bit-exact.

## Checks

| Check | Result |
| --- | --- |
| `action15.required_files` | PASS |
| `action15.episode_index_type` | PASS |
| `action30.required_files` | PASS |
| `action30.episode_index_type` | PASS |
| `action15.info_counts` | PASS |
| `action15.feature_keys` | PASS |
| `action15.feature_schema` | PASS |
| `action15.profile_contract` | PASS |
| `action15.task_instruction` | PASS |
| `action15.episode_boundaries` | PASS |
| `action30.info_counts` | PASS |
| `action30.feature_keys` | PASS |
| `action30.feature_schema` | PASS |
| `action30.profile_contract` | PASS |
| `action30.task_instruction` | PASS |
| `action30.episode_boundaries` | PASS |
| `cross_profile.metadata_counts_identity` | PASS |
| `cross_profile.main_row_count` | PASS |
| `cross_profile.main_float32_bit_identity` | PASS |
| `cross_profile.default_columns_identity` | PASS |
| `main.default_column_semantics` | PASS |
| `cross_profile.episode_index_json_identity` | PASS |
| `episode_00.shared_sidecar_identity` | PASS |
| `episode_00.anchor_map` | PASS |
| `episode_00.qpos_audit` | PASS |
| `episode_00.action30_timestamps` | PASS |
| `episode_00.endpoint_float32_bit_identity` | PASS |
| `episode_00.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_00.record_counts` | PASS |
| `sidecars.aggregate_counts` | PASS |
| `partial_suffix_contract` | PASS |
| `action15.standard_stats` | PASS |
| `action15.effective_pi0_stats` | PASS |
| `action15.normalization_roundtrip` | PASS |
| `action30.standard_stats` | PASS |
| `action30.effective_pi0_stats` | PASS |
| `action30.normalization_roundtrip` | PASS |
| `cross_profile.effective_state_stats_identity` | PASS |
| `action30.carrier_sidecar_stats` | PASS |
| `cross_profile.monitor_file_identity` | PASS |
| `monitor.valid_training_anchor_subset` | PASS |
| `monitor.provenance_hashes` | PASS |
| `action30.fail_fast_profile_marker` | PASS |
| `action15.lerobot_dataset_counts` | PASS |
| `action15.random_and_boundary_video_decode` | PASS |
| `action30.lerobot_dataset_counts` | PASS |
| `action30.random_and_boundary_video_decode` | PASS |
| `cross_profile.decoded_video_bit_identity` | PASS |
| `video.raw_mcap_lossy_pixel_qa` | PASS |
