# FastWAM training fixture：一次发送、一次接收

这套 fixture 用于比较同一条训练 observation 在离线数据链路与真实部署 server 链路中的输入。
所有可提交文件都位于 `lerobot/franka_project`；client 拉取 `lerobot` 仓库后即可回放，不需要
FastWAM checkout、训练 dataset 或原始 MCAP。

## 已冻结的样本

- fixture：`fixtures/async/fastwam_grab_cups_observation_v1.npz`
- provenance ledger：`fixtures/async/fastwam_grab_cups_observation_v1.json`
- task：`grab the paper cup.`
- 训练 dataset：`franka_eef_grab_cups_v2`
- output episode/frame/global index：`4 / 106 / 935`
- raw source：`run_20260725_124233_00108`
- raw indices：camera1 `107`、camera2 `107`、EEF `106`、gripper `1793`

这个 anchor 后面的 32-step 训练窗口有明显的夹取并抬升：第 12 个 action 的
`gripper.open_target_0_1` 首次不高于 `0.2`，最小值为 `0.0176074`；EEF z 从
`0.203414 m` 上升到 `0.278357 m`，净上升 `0.0749433 m`。

NPZ 严格保持 client wire schema，且不包含训练 action：

```text
state    float32 [10]
camera1  uint8   [480, 640, 3]
camera2  uint8   [480, 640, 3]
```

FastWAM 训练 dataset 中的视频已经是 `224×224` H.264，不能直接满足 client 的
`480×640` wire contract。因此 exporter 根据 conversion ledger 回溯到同一个 raw MCAP
observation。验证结果记录在 JSON ledger 中：

- raw 10D wire state 经 server 同语义转换后，与训练 8D proprio 最大绝对误差
  `1.2870e-7`；阈值为 `2e-6`；
- 两路 raw 图像经过部署侧 center-crop + bilinear resize 后，与 conversion-time
  pre-encode 结果逐像素完全一致；
- pre-encode 与训练 H.264 解码图的合并 MAE 为 `2.0593/255`，PSNR 为 `39.0145 dB`；
  这是视频编码残差，不是 crop 几何差异。

## 1. 启动 FastWAM server

在有 checkpoint/GPU 的 server 机器上，从 `lerobot` checkout 启动：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

CUDA_VISIBLE_DEVICES=1 python franka_project/scripts/serve_franka_pi0_async.py \
  --host=127.0.0.1 \
  --port=15173 \
  --fps=30 \
  --inference_latency=0 \
  --obs_queue_timeout=1 \
  --observation_similarity_mode=none \
  --policy_type=fastwam \
  --pretrained_name_or_path=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_grab_cups_v2/checkpoints/weights/step_030750.pt \
  --actions_per_chunk=32 \
  --policy_device=cuda \
  --fastwam_joint_video_inference=true \
  --fastwam_joint_video_output_dir=/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/depoly_joint_videos
```

如果 client 与 server 在同一台机器且没有 tunnel，后续把地址设成
`127.0.0.1:15173`。如果沿用现有 `8080 -> tunnel -> 15173` 拓扑，则使用默认的
`127.0.0.1:8080`。

## 2. client 只发送一次 fixture

在 client 机器拉取包含 fixture 的 `lerobot` 提交后执行：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

python franka_project/scripts/send_async_fixture_once.py \
  --server-address=127.0.0.1:8080 \
  --fixture="$PWD/franka_project/fixtures/async/fastwam_grab_cups_observation_v1.npz" \
  --output=/tmp/fastwam_fixture_response.json \
  --task='grab the paper cup.' \
  --timeout-s=120
```

这个入口固定执行以下流程：

1. 校验同名 JSON ledger、NPZ SHA-256、task、30 Hz、32-step 和 `rename_map={}`；
2. 用 `FrankaRosConfig(dry_run=True)` 读取 fixture；
3. 完成 `PolicySetupAck` 握手；
4. 只发送一个 `must_go=True` observation；
5. 只接收一个 32-step absolute8 action chunk；
6. 原子写入 `/tmp/fastwam_fixture_response.json`，随后 ACK 并退出。

脚本不会把返回 action 放入控制 queue，不会 import ROS2 runtime，也不会发布 controller
命令；输出 JSON 中固定写有 `"actions_executed": false`。`--action-log` 对应 dry-run backend
要求的本地 sink，但该一次性入口不调用 `robot.send_action()`，因此不会向其中写动作。

查看预测：

```bash
jq '{fixture_sha256, observation_timestep, source_timestep, actions_executed,
     first_action: .actions[0], last_action: .actions[-1]}' \
  /tmp/fastwam_fixture_response.json
```

若 server 直接监听在本机 `15173`，只改这一项：

```bash
python franka_project/scripts/send_async_fixture_once.py \
  --server-address=127.0.0.1:15173
```

## 3. 重新生成或换选帧

正常 client 回放不需要执行本节。只有能访问 FastWAM dataset 和原始 MCAP 的数据机器才能
重新生成：

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

/ytech_milm_intern/tujiahang/miniconda3/envs/lerobot/bin/python \
  franka_project/scripts/export_fastwam_async_fixture.py \
  --fastwam-root=/m2v_intern/tujiahang/Projects/FastWAM \
  --episode-index=4 \
  --frame-index=106 \
  --output="$PWD/franka_project/fixtures/async/fastwam_grab_cups_observation_v1.npz"
```

`--frame-index` 必须还能提供完整的 33 observation / 32 action 训练窗口；否则 exporter
fail closed。它还会拒绝 dataset profile、state/action names、相机 topic/encoding、causal
alignment、crop、FPS、chunk size 或 task 的任何契约漂移。

## 4. 提交前验证

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

PYTHONPATH="$PWD/franka_project/src:$PWD/franka_project/ros_lerobot/src" \
  python -m pytest -q -p no:cacheprovider \
  franka_project/tests/test_export_fastwam_async_fixture.py
```

需要提交的相关文件是：

```text
.gitignore
franka_project/markdown/FASTWAM_FIXTURE.md
franka_project/markdown/PI0_FASTWAM_LAUNCH_GUIDE.md
franka_project/scripts/export_fastwam_async_fixture.py
franka_project/scripts/send_async_fixture_once.py
franka_project/fixtures/async/fastwam_grab_cups_observation_v1.npz
franka_project/fixtures/async/fastwam_grab_cups_observation_v1.json
franka_project/tests/test_export_fastwam_async_fixture.py
```
