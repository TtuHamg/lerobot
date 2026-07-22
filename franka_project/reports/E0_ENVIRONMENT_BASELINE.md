# E0 Frank3 环境基线

## 结论

E0 通过。用户指定的 Conda `lerobot` 环境可完整解码 Frank3 MCAP，两张 A800 的 BF16/cuDNN 前反向和 NCCL 通信均通过，本地 `lerobot/pi0_base` 权重完整可读。W&B 在线记录已验证，smoke run 仅上传了一个标量，没有上传 artifact 或 model checkpoint。

## 固定启动方式

```bash
source /ytech_milm_intern/tujiahang/.bashrc >/dev/null 2>&1
conda activate lerobot
unset LD_LIBRARY_PATH
```

`unset LD_LIBRARY_PATH` 是必须的：当前 shell 继承的系统 cuDNN 9.11 与 PyTorch 2.11 所需 cuDNN 9.19 冲突。清除后，两张 GPU 的 BF16 convolution forward/backward 均为 finite。

## 软件与模型

| 项目 | 版本/值 |
|---|---|
| Python | 3.12.13 |
| LeRobot | 0.5.2（editable workspace） |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA / cuDNN / NCCL | 12.8 / 9.19.0 / 2.28.9 |
| Transformers / datasets / accelerate | 5.5.4 / 4.8.5 / 1.14.0 |
| W&B | 0.27.2 |
| MCAP / ROS2 support | 1.4.0 / 0.5.7 |
| ffmpeg / ffprobe | 7.1.1 / 7.1.1 |
| 仓库 commit | `a869295ffadf3e82054fb8f6abdd7d31551ac55f` |
| `uv.lock` SHA256 | `af48fc72356c996f4f2b746069274a373884f4e0239724ffd811f4ebcc2d84e8` |

为本次工作向已授权 Conda 环境补充了 `mcap==1.4.0`、`mcap-ros2-support==0.5.7`、`pytest==8.4.2`、`lz4==4.4.5` 和 `pypdf==6.5.0`。本机没有 `uv` 入口，所以实验命令均用上述 Conda 前缀。

`lerobot/pi0_base` 本地 revision 为 `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`，逻辑大小约 14.01 GB；配置的 `chunk_size=50`、`max_state_dim=32`、`max_action_dim=32`，可容纳当前 10D state/7D action。

## GPU 和 I/O smoke

| 检查 | 结果 |
|---|---|
| GPU | 2 × NVIDIA A800-SXM4-80GB，快照时每张仅占 2 MiB |
| BF16 | 两张均 supported |
| cuDNN | BF16 Conv2d forward/backward 通过，output/gradient 均 finite |
| NCCL | 两 rank 输入 1 和 2，all-reduce 后均为 3 |
| 官方 loader | `run_20260714_212901_00063` 完整解码，T=588 |
| 双目图像 | 两路均 `[588,480,640,3] uint8` |
| state | qpos `[588,7]`、gripper `[588,1]`、EEF `[588,7]` |

可重跑的 GPU 脚本为 `environment/e0_cuda_nccl_smoke.py`，SHA256 为 `7338d9f1e1dc98f03238fd60ed5ac24069ac116bebd78d3794a2d2fdcd5ade81`。

## W&B 策略验证

- entity/project：`ttuhamg/franka_pi0_full_eef`
- run：[`syujv5nl`](https://wandb.ai/ttuhamg/franka_pi0_full_eef/runs/syujv5nl)
- 状态：`finished`
- 历史值：`e0/smoke_scalar=1`
- `logged_artifact_count=0`
- 没有 model checkpoint 上传

后续训练可用 W&B 记录 train/validation loss 和系统指标，但配置中不调用 `wandb.Artifact`/checkpoint upload。

## 可复现性与限制

机器可读环境快照为 `environment/e0_environment_manifest.json`。仓库当时已有用户的 dirty/untracked 内容，本任务保留了无关改动；快照中同时记录了 commit、`uv.lock` hash 和 dirty-state hash。项目文件系统剩余约 251.8 TB，raw 数据文件系统剩余约 2.00 PB，E0 未见存储容量阻断。

