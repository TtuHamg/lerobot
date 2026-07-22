# D2 Cartesian Pipeline Spike

- Phase status: **PASS**
- Generated at: `2026-07-15T11:02:53.632878+00:00`
- Scope: two raw MCAP episodes; numeric EEF/gripper/timestamp paths only.
- Explicitly not performed: image conversion, full dataset conversion, PI0 training.

## Episode and full-horizon coverage

| Episode | Camera anchors | Candidate anchors | Valid K=50 | Valid K=100 | Endpoint max abs diff |
|---|---:|---:|---:|---:|---:|
| run_20260714_212901_00063 | 587 | 537 | 468 | 468 | 0.000e+00 |
| run_20260714_211745_00057 | 406 | 356 | 356 | 356 | 0.000e+00 |

A valid anchor contains the current observation plus the complete future 50-camera-interval horizon; no action padding is used. The 30 Hz layout is midpoint, endpoint for every camera interval, so its endpoint slots (`1::2` in zero-based Python indexing) are copied exactly from the 15 Hz carrier.

## Geometry and normalization acceptance

| Profile | State shape | Action shape | Position error (m) | Rotation error (rad) | Gripper error | Max theta (rad) | State norm RT | Action norm RT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| action15 | 824×10 | 824×50×7 | 0.000e+00 | 5.162e-08 | 0.000e+00 | 0.979978 | 8.327e-17 | 1.110e-16 |
| action30 | 824×10 | 824×100×7 | 0.000e+00 | 5.162e-08 | 0.000e+00 | 0.981708 | 8.327e-17 | 1.110e-16 |

All model-visible arrays and statistics are finite. State is 10D (`[xyz, rotation6D, gripper]`) and action is 7D (`[delta xyz, body-frame rotation vector, gripper]`).

Acceptance thresholds:

- Position encode/decode: `< 1.0e-06 m`
- Rotation encode/decode: `< 1.0e-05 rad`
- Gripper encode/decode: `< 1.0e-12`
- SO(3) branch margin: `theta < pi - 0.1`
- Normalize/unnormalize: `< 1.0e-10`

## Reproducibility hashes

- Manifest SHA-256: `d2ba4dc819eafd0163ed6673f53ea6e6cc7d492b075978c4db925c5b68dcf38b`
- Combined config SHA-256: `bdfc3eeaab65e47d64d171a1a0fad77657691a13a81baed8eaf2d9a9a6a22300`
- Pipeline source SHA-256: `bbd0aa8d575de622a65bb604ab60a3936e61d6fc27d977eae701d1021d937e41`
- Two-episode source dataset SHA-256: `55d353d52b54c0b7f84e96b91946a577b069d97cc6d3ac56a01a9f0fb417c426`

Individual config, source-code, and raw MCAP hashes are recorded in the machine-readable JSON.

## Outputs and boundary

- Machine report: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/conversion/d2_cartesian_pipeline_spike.json`
- 15 Hz stats: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/stats/pi0_eef_stats_spike_action15.json`
- 30 Hz stats: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/stats/pi0_eef_stats_spike_action30.json`

D2 passes for these two available episodes. This result does not authorize D3: it neither resolves missing raw episodes in D0 nor creates/trains on a full dataset.
