# D3 Full Dataset Conversion (22of25)

- Status: **PASS_PARTIAL_SMOKE**
- Generated: `2026-07-15T12:36:33.945474+00:00`
- Source episodes: **1**
- Common camera anchors: **587**
- Main rows per dataset: **586**
- Valid full-horizon anchors: **468**
- 30 Hz carrier rows: **1172**

Both datasets use the 15 Hz camera clock. The 15 Hz main-table `action` and the 30 Hz sidecar are absolute 8D audit carriers; PI0 must use the project Cartesian adapter and the model-visible 10D/7D effective statistics.

- 15 Hz dataset: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/.cache/d3_smoke/franka_current_eef_obs15_act15_v1_22of25_76b839b2_partial1`
- 30 Hz dataset: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/.cache/d3_smoke/franka_current_eef_obs15_act30_v1_22of25_76b839b2_partial1`
- Machine report: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/conversion/d3_conversion_partial1.json`

No validation/test split was created. The frozen monitor subset is sampled from the same training anchors and is only an in-distribution fitting diagnostic.
