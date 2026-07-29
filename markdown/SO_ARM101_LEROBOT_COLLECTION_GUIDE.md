# SO-ARM101 + LeRobot 数据采集与训练流程

这份文档记录当前 SO-ARM101 / SO101 leader-follower 机械臂配合 LeRobot 的常用流程，包括串口权限、校准、遥操作、带相机预览、数据采集、数据集可视化、ACT 训练和 rollout。

## 设备约定

当前命令默认：

```text
follower arm: /dev/ttyACM1, id=tjh_follower_arm
leader arm:   /dev/ttyACM0, id=tjh_leader_arm
front camera: OpenCV index 0
wrist camera: OpenCV index 2
```

如果 USB 重新插拔，`/dev/ttyACM0` 和 `/dev/ttyACM1` 可能会变化。运行前建议确认端口。

## 1. 串口权限

每次重新插拔设备或重启机器后，如果没有串口权限，可以执行：

```bash
sudo chmod 666 /dev/ttyACM*
```

如果 `/dev/ttyACM*` 不存在，说明设备没有识别到，先检查 USB 连接和供电。

## 2. 校准机械臂

### 校准 follower arm

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tjh_follower_arm
```

### 校准 leader arm

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tjh_leader_arm
```

注意：这里 leader arm 仍使用 `so101_follower` 类型进行校准，是当前命令中的用法。后续遥操作时 leader 使用 `--teleop.type=so101_leader`。

## 3. 基础遥操作测试

不启用相机，只测试 leader 控制 follower：

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tjh_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tjh_leader_arm
```

如果 follower 不动，优先检查：

```text
1. follower 和 leader 端口是否写反
2. 两个机械臂是否都完成校准
3. 是否有其他进程占用 /dev/ttyACM*
4. 电机是否供电，是否有 Torque_Enable 通信错误
```

## 4. 带相机预览的遥操作

用于确认相机 index、画面方向、机械臂控制是否正常。

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tjh_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tjh_leader_arm \
    --display_data=true
```

说明：

```text
front      -> 前视相机
wrist.left -> 腕部相机
```

如果某个相机打不开，通常是 `index_or_path` 不对，或者相机被其他进程占用。

## 5. 数据采集

### 任务 0：把黑色瓶盖放入白色纸杯

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tjh_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tjh_leader_arm \
    --display_data=true \
    --dataset.repo_id=tuuy/lerobot_so_arm101_task0_new \
    --dataset.num_episodes=25 \
    --dataset.single_task="Place the black bottle cap into the white paper cup" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=5
```

含义：

```text
num_episodes=25       采集 25 条 episode
episode_time_s=25     每条 episode 25 秒
reset_time_s=5        每条 episode 后给 5 秒复位时间
push_to_hub=false     只保存在本地，不上传 Hub
```

### 任务 1：把黑色瓶盖放到白色纸杯右侧

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tjh_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tjh_leader_arm \
    --display_data=true \
    --dataset.repo_id=tuuy/lerobot_so_arm101 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Place the black bottle cap to the right of the white paper cup" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --resume=true \
    --dataset.root=/home/tjh/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_20260617_193601
```

这里使用了：

```text
--resume=true
```

表示在已有数据集目录上继续采集。`--dataset.root` 必须指向之前那份本地数据集目录。

## 6. 数据集可视化

示例：

```bash
WGPU_BACKEND=vulkan lerobot-dataset-viz \
    --repo-id tuuy/lerobot_so_arm101 \
    --root /home/tjh/.cache/huggingface/lerobot \
    --mode local \
    --episode-index=2
```

注意：原始命令中的 root 是：

```text
/home/tjh/.cache/huggingface/lero
```

这里更可能应该是：

```text
/home/tjh/.cache/huggingface/lerobot
```

如果可视化找不到数据集，优先检查 `--root` 是否写对。

## 7. ACT 策略训练

以下命令使用任务 0 数据集训练 ACT：

```bash
lerobot-train \
  --dataset.repo_id=tuuy/lerobot_so_arm101_task0_new \
  --dataset.root=/home/tjh/.cache/huggingface/lerobot/tuuy/lerobot_so_arm101_task0_new_20260618_112610 \
  --dataset.revision=v0.4.0 \
  --dataset.streaming=false \
  --policy.type=act \
  --dataset.video_backend=pyav \
  --output_dir=output_lerobot_train/act/task0_new \
  --job_name=cap_place_job \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=tjh_lerobot_act_task0_new \
  --policy.push_to_hub=false \
  --steps=300000 \
  --batch_size=4
```

关键参数：

```text
policy.type=act       使用 ACT 策略
steps=300000          训练 30 万 step
batch_size=4          batch size 为 4
video_backend=pyav    使用 pyav 解码视频
wandb.enable=true     启用 wandb 记录
```

如果显存不足，可以优先降低：

```text
--batch_size
```

## 8. Rollout / 策略实机测试

使用训练好的 ACT checkpoint 在 follower 上执行策略：

```bash
lerobot-rollout \
    --strategy.type=episodic \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist.left: {type: opencv, index_or_path: 2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}' \
    --robot.id=tjh_follower_arm \
    --display_data=true \
    --policy.path=/home/tjh/Projects/lerobot/output_lerobot_train/act/task0/checkpoints/180000/pretrained_model \
    --dataset.repo_id=tuuy/rollout_lerobot_so_arm101_single \
    --dataset.single_task="Place the black bottle cap into the white paper cup" \
    --dataset.episode_time_s=1000 \
    --dataset.reset_time_s=5 \
    --dataset.num_episodes=1 \
    --dataset.push_to_hub=false
```

注意：

```text
--policy.path
```

需要指向实际 checkpoint 目录。测试不同 checkpoint 时只需要替换这个路径。

## 9. 推荐检查顺序

每次采集或 rollout 前建议按顺序确认：

```text
1. sudo chmod 666 /dev/ttyACM*
2. 确认 follower 是 /dev/ttyACM1，leader 是 /dev/ttyACM0
3. 确认两个机械臂校准文件 id 正确
4. 先跑不带相机的 lerobot-teleoperate
5. 再跑带相机的 lerobot-teleoperate --display_data=true
6. 确认画面、遥操作、夹爪都正常后再 lerobot-record
7. 采集后用 lerobot-dataset-viz 抽查 episode
8. 训练前确认 dataset.root 指向实际本地数据集目录
9. rollout 前确认 checkpoint 路径和相机 key 与训练时一致
```

