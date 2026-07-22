# D4 Franka LeRobot dataset verification

- Status: **PASS_FULL_VERIFICATION**
- Generated: `2026-07-15T12:52:28.012238+00:00`
- Config: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/configs/data/franka_current_eef_conversion_22of25_v1.yaml`
- Action-15 root: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2`
- Action-30 root: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/data/lerobot/franka_current_eef_obs15_act30_v1_22of25_76b839b2`

## Verified counts

```json
{
  "episodes": 22,
  "common_camera_anchors": 8815,
  "main_rows": 8793,
  "valid_anchors": 7465,
  "action15_effective_stats_count": 373250,
  "action30_carrier_rows": 17586,
  "action30_effective_stats_count": 746500
}
```

## Video QA

- Random/boundary decoded samples: [0, 585, 1790, 4584, 4822, 5190, 7460, 8534, 8792]
- Raw-MCAP comparisons: 18
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
| `episode_01.shared_sidecar_identity` | PASS |
| `episode_01.anchor_map` | PASS |
| `episode_01.qpos_audit` | PASS |
| `episode_01.action30_timestamps` | PASS |
| `episode_01.endpoint_float32_bit_identity` | PASS |
| `episode_01.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_01.record_counts` | PASS |
| `episode_02.shared_sidecar_identity` | PASS |
| `episode_02.anchor_map` | PASS |
| `episode_02.qpos_audit` | PASS |
| `episode_02.action30_timestamps` | PASS |
| `episode_02.endpoint_float32_bit_identity` | PASS |
| `episode_02.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_02.record_counts` | PASS |
| `episode_03.shared_sidecar_identity` | PASS |
| `episode_03.anchor_map` | PASS |
| `episode_03.qpos_audit` | PASS |
| `episode_03.action30_timestamps` | PASS |
| `episode_03.endpoint_float32_bit_identity` | PASS |
| `episode_03.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_03.record_counts` | PASS |
| `episode_04.shared_sidecar_identity` | PASS |
| `episode_04.anchor_map` | PASS |
| `episode_04.qpos_audit` | PASS |
| `episode_04.action30_timestamps` | PASS |
| `episode_04.endpoint_float32_bit_identity` | PASS |
| `episode_04.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_04.record_counts` | PASS |
| `episode_05.shared_sidecar_identity` | PASS |
| `episode_05.anchor_map` | PASS |
| `episode_05.qpos_audit` | PASS |
| `episode_05.action30_timestamps` | PASS |
| `episode_05.endpoint_float32_bit_identity` | PASS |
| `episode_05.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_05.record_counts` | PASS |
| `episode_06.shared_sidecar_identity` | PASS |
| `episode_06.anchor_map` | PASS |
| `episode_06.qpos_audit` | PASS |
| `episode_06.action30_timestamps` | PASS |
| `episode_06.endpoint_float32_bit_identity` | PASS |
| `episode_06.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_06.record_counts` | PASS |
| `episode_07.shared_sidecar_identity` | PASS |
| `episode_07.anchor_map` | PASS |
| `episode_07.qpos_audit` | PASS |
| `episode_07.action30_timestamps` | PASS |
| `episode_07.endpoint_float32_bit_identity` | PASS |
| `episode_07.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_07.record_counts` | PASS |
| `episode_08.shared_sidecar_identity` | PASS |
| `episode_08.anchor_map` | PASS |
| `episode_08.qpos_audit` | PASS |
| `episode_08.action30_timestamps` | PASS |
| `episode_08.endpoint_float32_bit_identity` | PASS |
| `episode_08.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_08.record_counts` | PASS |
| `episode_09.shared_sidecar_identity` | PASS |
| `episode_09.anchor_map` | PASS |
| `episode_09.qpos_audit` | PASS |
| `episode_09.action30_timestamps` | PASS |
| `episode_09.endpoint_float32_bit_identity` | PASS |
| `episode_09.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_09.record_counts` | PASS |
| `episode_10.shared_sidecar_identity` | PASS |
| `episode_10.anchor_map` | PASS |
| `episode_10.qpos_audit` | PASS |
| `episode_10.action30_timestamps` | PASS |
| `episode_10.endpoint_float32_bit_identity` | PASS |
| `episode_10.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_10.record_counts` | PASS |
| `episode_11.shared_sidecar_identity` | PASS |
| `episode_11.anchor_map` | PASS |
| `episode_11.qpos_audit` | PASS |
| `episode_11.action30_timestamps` | PASS |
| `episode_11.endpoint_float32_bit_identity` | PASS |
| `episode_11.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_11.record_counts` | PASS |
| `episode_12.shared_sidecar_identity` | PASS |
| `episode_12.anchor_map` | PASS |
| `episode_12.qpos_audit` | PASS |
| `episode_12.action30_timestamps` | PASS |
| `episode_12.endpoint_float32_bit_identity` | PASS |
| `episode_12.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_12.record_counts` | PASS |
| `episode_13.shared_sidecar_identity` | PASS |
| `episode_13.anchor_map` | PASS |
| `episode_13.qpos_audit` | PASS |
| `episode_13.action30_timestamps` | PASS |
| `episode_13.endpoint_float32_bit_identity` | PASS |
| `episode_13.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_13.record_counts` | PASS |
| `episode_14.shared_sidecar_identity` | PASS |
| `episode_14.anchor_map` | PASS |
| `episode_14.qpos_audit` | PASS |
| `episode_14.action30_timestamps` | PASS |
| `episode_14.endpoint_float32_bit_identity` | PASS |
| `episode_14.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_14.record_counts` | PASS |
| `episode_15.shared_sidecar_identity` | PASS |
| `episode_15.anchor_map` | PASS |
| `episode_15.qpos_audit` | PASS |
| `episode_15.action30_timestamps` | PASS |
| `episode_15.endpoint_float32_bit_identity` | PASS |
| `episode_15.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_15.record_counts` | PASS |
| `episode_16.shared_sidecar_identity` | PASS |
| `episode_16.anchor_map` | PASS |
| `episode_16.qpos_audit` | PASS |
| `episode_16.action30_timestamps` | PASS |
| `episode_16.endpoint_float32_bit_identity` | PASS |
| `episode_16.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_16.record_counts` | PASS |
| `episode_17.shared_sidecar_identity` | PASS |
| `episode_17.anchor_map` | PASS |
| `episode_17.qpos_audit` | PASS |
| `episode_17.action30_timestamps` | PASS |
| `episode_17.endpoint_float32_bit_identity` | PASS |
| `episode_17.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_17.record_counts` | PASS |
| `episode_18.shared_sidecar_identity` | PASS |
| `episode_18.anchor_map` | PASS |
| `episode_18.qpos_audit` | PASS |
| `episode_18.action30_timestamps` | PASS |
| `episode_18.endpoint_float32_bit_identity` | PASS |
| `episode_18.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_18.record_counts` | PASS |
| `episode_19.shared_sidecar_identity` | PASS |
| `episode_19.anchor_map` | PASS |
| `episode_19.qpos_audit` | PASS |
| `episode_19.action30_timestamps` | PASS |
| `episode_19.endpoint_float32_bit_identity` | PASS |
| `episode_19.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_19.record_counts` | PASS |
| `episode_20.shared_sidecar_identity` | PASS |
| `episode_20.anchor_map` | PASS |
| `episode_20.qpos_audit` | PASS |
| `episode_20.action30_timestamps` | PASS |
| `episode_20.endpoint_float32_bit_identity` | PASS |
| `episode_20.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_20.record_counts` | PASS |
| `episode_21.shared_sidecar_identity` | PASS |
| `episode_21.anchor_map` | PASS |
| `episode_21.qpos_audit` | PASS |
| `episode_21.action30_timestamps` | PASS |
| `episode_21.endpoint_float32_bit_identity` | PASS |
| `episode_21.valid_anchors_and_t_plus_1_t_plus_50` | PASS |
| `episode_21.record_counts` | PASS |
| `sidecars.aggregate_counts` | PASS |
| `full_22of25_expected_counts` | PASS |
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
