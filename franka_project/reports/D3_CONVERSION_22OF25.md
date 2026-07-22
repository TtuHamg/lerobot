# D3 Full Dataset Conversion (22of25)

- Status: **PASS_FULL_CONVERSION**
- Generated: `2026-07-15T12:50:56.002834+00:00`
- Source episodes: **22**
- Common camera anchors: **8815**
- Main rows per dataset: **8793**
- Valid full-horizon anchors: **7465**
- 30 Hz carrier rows: **17586**

Both datasets use the 15 Hz camera clock. The 15 Hz main-table `action` and the 30 Hz sidecar are absolute 8D audit carriers; PI0 must use the project Cartesian adapter and the model-visible 10D/7D effective statistics.

- 15 Hz dataset: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/data/lerobot/franka_current_eef_obs15_act15_v1_22of25_76b839b2`
- 30 Hz dataset: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/data/lerobot/franka_current_eef_obs15_act30_v1_22of25_76b839b2`
- Machine report: `/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/conversion/d3_conversion_22of25.json`

No validation/test split was created. The frozen monitor subset is sampled from the same training anchors and is only an in-distribution fitting diagnostic.
