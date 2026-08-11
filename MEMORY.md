# Franka + FastWAM 真机部署 Memory

> 最后更新：2026-08-11
> 适用范围：pnp 真机 client、KML-dev FastWAM server、Franka FCI/ROS 2 控制链路。
> 这是部署不变量和事故复盘，不替代启动指南。改动相关代码前，先读本文并核对文末引用。

## 1. 当前拓扑与模型

- 真机 client：`pnp@172.22.114.15`
- client 仓库：`/home/pnp/Projects/lerobot`
- Franka 控制脚本与 ROS 2 workspace：`/home/pnp/franka`
- FastWAM server：`KML-dev`，SSH 为 `root@127.0.0.1:2222`
- server 仓库：`/m2v_intern/tujiahang/Projects/lerobot`
- 当前 mix3 checkpoint：
  `/m2v_intern/tujiahang/Projects/FastWAM/franka_project/runs/franka_eef_mix3_0804_joint_lora_after_warmup_pretrained_xt/lora_after_warmup_bs256_lr1e-4_r32a64/checkpoints/weights/step_005000.pt`

当前为了调试，client **有意等一个 chunk 执行完再请求/执行下一个 chunk**。这会降低闭环响应速度，但不是下述 frame 或 gripper contract 的修复点；以后改为滚动执行时，不要顺带改变 pose/gripper 语义。

## 2. 必须保持的部署契约

| 项目 | 当前正确契约 | 禁止的隐式假设 |
|---|---|---|
| mix3 pose | checkpoint 的 observation/action pose 语义是 `O_T_LINK8` | 字段名含 `eef` 就等于物理 TCP/EEF |
| ROS action chunk | 下发绝对 `O_T_EEF` | 直接把模型的 `O_T_LINK8` 当作 `O_T_EEF` |
| pose 转换 | 使用完整 SE(3)：`O_T_EEF = O_T_LINK8 * F_T_EE` | 只旋转 quaternion，或直接加减平移偏置 |
| model gripper | state 和 action 均按兼容模式 `closed_0_1` | 根据字段旧名字猜成 `open_0_1`，或在多层重复取反 |
| canonical gripper | `0 = 全开`，`1 = 全闭` | 把 `0.98` 当 openness |
| 物理夹爪端点 | 从 `/franka_gripper_follower` 参数一次读取并冻结；当前为 `0.0/0.8 rad` | 从当前 `/gripper/joint_states` 样本推断端点，或继续使用旧 `0.4` 上限 |
| controller inactive | 通常是 FCI/reflex 后的结果 | 把 inactive 当成第一根因并直接反复 activate |

KML-dev 启动 FastWAM server 时，这个已训练 checkpoint 必须显式带上：

```bash
--fastwam_state_gripper_encoding=closed_0_1 \
--fastwam_action_gripper_encoding=closed_0_1
```

pnp 上启动 client 时必须带上：

```bash
--robot.policy_eef_frame=link8
```

## 3. FCI 环掉与 controller `inactive`

### 3.1 看到的现象

典型顺序是：

1. libfranka 报 `motion aborted by reflex` 和 `communication_constraints_violation`；
2. Franka hardware interface 被 deactive/unconfigure；
3. `joint_impedance_ik_controller`、`franka_robot_state_broadcaster` 等显示 `inactive`；
4. gateway 必须进入 HOLD，禁止继续 arm 或发送动作。

所以 **controller inactive 是 FCI 控制环已经失败后的结果，不是第一根因**。先看 arm controller 日志中的第一条 FCI/ControlException，再决定恢复方式。

### 3.2 已确认的原因

这里先后发现了两个实时性风险：

1. 旧部署把整个 `ros2_control_node` 固定在单个 CPU 10 上，FCI 实时线程与 DDS/ROS 非实时线程争抢同一个核。现在启动脚本给进程 CPU 9-10，并按线程拆分：实时调度线程在 CPU 10，普通/DDS 线程在 CPU 9；网卡 IRQ 放 CPU 8。
2. **重复触发问题的决定性根因**在 `franka_robot_state_broadcaster`：1 kHz `update()` 会等待 `publish_mutex_`，而发布线程拿着该锁连续执行 8 次 DDS publish。DDS 背压时，实时 update 被阻塞，曾观测到 broadcaster 最大执行时间约 `318448 us`、hardware read 周期约 `27083 us`、IK controller 周期约 `22964 us`，远超 1 ms FCI 预算。

最终修复位于：

- `/home/pnp/franka/franka_ros2_ws/src/franka_robot_state_broadcaster/src/franka_robot_state_broadcaster.cpp`
- `/home/pnp/franka/franka_ros2_ws/src/franka_robot_state_broadcaster/test/test_franka_robot_state_broadcaster.cpp`

修复原则：

- 1 kHz `update()` 使用 `std::try_to_lock`；拿不到发布锁就只跳过本帧 telemetry，绝不等待。
- 发布线程只在锁内复制 snapshot，所有 DDS publish 都在锁外执行。
- snapshot 被取走后设置 `has_fresh_state_ = false`，不重复发布旧状态。
- 单元测试覆盖“发布线程占锁时 update 不等待”。

这不会让 30 Hz client 获得“错误时间戳拼接”的状态：丢的是一帧非关键 telemetry，下一次拿到锁会发布一份完整、同一 snapshot 的新状态。它比阻塞 1 kHz FCI 环安全得多。client 原有的新鲜度/时间戳检查仍必须保留。

人工制造 DDS 背压 40 秒后的验证结果：broadcaster 最大约 `213.7 us`，hardware/IK 周期最大约 `1.45/1.41 ms`，controller 保持 active，未再出现 communication fault。

### 3.3 CPU 与启动脚本的配套修复

`/home/pnp/franka/start_real_validation.sh` 当前要点：

- 启动传入 `fci_cpu_affinity:=9-10`；
- 运行后将实时调度线程约束到 CPU 10、普通/DDS 线程约束到 CPU 9；
- 不再让旧的 `/usr/local/sbin/franka-fci-pin.sh` 或 systemd slice `AllowedCPUs=10` 把整个进程重新压回单核；
- 外层 launch lock 使用 `flock --close`，避免锁 fd 被子进程继承导致误判已有实例。

不要只做 CPU 隔离而回退 broadcaster 非阻塞修复；两者解决的是不同层面的风险。

### 3.4 诊断与安全恢复

```bash
source /opt/ros/jazzy/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
source /home/pnp/franka/haply_ros/install/setup.bash

ros2 control list_controllers
ros2 control list_hardware_components
ros2 topic echo /controller_manager/statistics/full --once
ros2 topic echo /lerobot/franka/safety_gateway_status --once
rg -n "communication_constraints_violation|ControlException|Deactivating Franka|motion aborted" \
  /home/pnp/franka/logs/real_validation/arm_controller.log
```

如果已经发生 FCI/reflex，不要在残留进程上反复 activate。先让 Franka Desk 完成必要的 acknowledge/unlock，然后干净重启：

```bash
bash /home/pnp/franka/stop_all.sh --keep-haply-manager
bash /home/pnp/franka/start_real_validation.sh
```

`stop_all.sh` 已包含 FrankaTeleop/Polymetis 清理。重启后系统应保持 HOLD、未 arm；确认 hardware/controller active 和 gateway ready 后，再由操作者明确 arm。

gateway 的 readiness 必须以 `/controller_manager/list_controllers` 和 hardware 状态轮询为准；controller 不是 active 时 `controller_ready=false`、arm fail closed、立即 HOLD。

## 4. EEF 与 link8 踩坑

### 4.1 根因

mix3 数据由 client 机器上的 FrankaTeleop 采集。历史字段/变量虽然称为 `current_pose` 或 EEF，但数值实际来自 `O_T_LINK8`。checkpoint 因此学到的是 link8/flange pose，而当前 ROS 控制接口接收的是物理 EEF/TCP pose。

训练 loss 收敛只说明网络拟合了训练 contract；如果真机部署把 `O_T_LINK8` 当 `O_T_EEF`，动作仍会因固定工具变换错误而严重偏离。

### 4.2 正确数据流

```text
ROS measured O_T_EEF
    -> O_T_LINK8 = O_T_EEF * inverse(F_T_EE)
    -> FastWAM observation / prediction（link8 contract）
    -> O_T_EEF = O_T_LINK8 * F_T_EE
    -> CartesianActionChunk（EEF contract）
    -> gateway 移除固定 F_T_EE 后对 link8 IK tip 求解
```

client 在 `--robot.policy_eef_frame=link8` 下，从
`/franka_robot_state_broadcaster/robot_state` 获取一次固定的 `F_T_EE`，然后立即注销这个 1 kHz subscription，避免给控制链额外施压。没有拿到 `F_T_EE` 时必须 fail fast，不能静默使用单位矩阵或把 link8 当 eef。

转换必须用齐次矩阵的完整 SE(3) 乘法。工具偏移的平移方向会随姿态旋转，只处理 rotation 或直接加减 xyz 都是错的。

关键实现：

- `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/config_franka_ros.py`
- `franka_project/ros_lerobot/src/lerobot_robot_franka_ros/ros2_runtime.py`
- `franka_project/ros2_ws/src/lerobot_franka_interfaces/msg/CartesianActionChunk.msg`
- `/home/pnp/franka/cartesian_ik_gateway.py`

启动日志应出现类似：

```text
F_T_EE captured for policy_eef_frame=link8; removed 1 kHz RobotState subscription
```

## 5. Gripper 语义与端点

### 5.1 已训练 checkpoint 的兼容方式

mix3 历史数据的字段命名、raw endpoint 与物理开合语义曾不一致。已训练 checkpoint 的 state/action wire 值实际应解释为 `closed_0_1`：

```text
closed_0_1 = 0  -> 全开
closed_0_1 = 1  -> 全闭
```

兼容旧 checkpoint 的正确方式是在 server 分别设置 state/action encoding override，而不是改 checkpoint、重写数据标签，或在 client/follower 再做一次 `1-x`。state 与 action 是两个独立入口，必须两个 flag 都核对。

`open_0_1` 的语义正好相反，只能用于确实按 openness 训练的 artifact。不要根据旧字段名选择 encoding，要依据 golden fixture/物理动作验证。

### 5.2 为什么给 0.98 却只闭合约 0.8

当时 follower 的物理 closed endpoint 仍是旧 `0.4 rad`，真实 Robotiq 2F-85 全闭端点约为 `0.8 rad`。因此 canonical closure `0.98` 只被映射到约 `0.392 rad`，视觉上像只闭了一半。

当前唯一映射为：

```text
joint_target = open_position
             + closed_0_1 * (closed_position - open_position)
```

本机当前参数应为：

```text
open_position   = 0.0
closed_position = 0.8
```

检查：

```bash
ros2 param get /franka_gripper_follower open_position
ros2 param get /franka_gripper_follower closed_position
```

live client 启动时从 `/franka_gripper_follower` 一次读取这两个 ROS 参数并冻结；参数服务不可用、值非有限或端点相同时必须启动失败。CLI 的 `robot.gripper_*_position` 只能作为配置/fallback，文档和命令也应写 `0.0/0.8`，避免误导。

不要从 `/gripper/joint_states` 动态学习端点：该 topic 表示当前状态，不是标定极限；机械臂在中间位置启动时会把中间值误当端点。`joint_states` 只用于按已冻结端点归一化当前 closure。

## 6. 首次运动剧烈晃动

首次动作曾因当前实测关节与第一条命令跨度大、速度限制过松且没有加速度/jerk 平滑而晃动。平滑应放在离实时 controller 最近、拥有实测 `q` 的 gateway 层，而不是让模型或网络 client 猜控制周期。

`/home/pnp/franka/cartesian_ik_gateway.py` 当前包含从 `q_meas` 出发的平滑启动、速度/加速度限制；接收新 chunk 时不能重置为模型首点。调参时保持 controller 在 HOLD，先验证小位移，不能靠降低 FCI 实时性要求来掩盖抖动。

## 7. 每次真机启动的 Go/No-Go

只有全部满足才允许 arm：

- 无残留 FrankaTeleop/Polymetis/旧 gateway/client 进程；
- Franka hardware 为 active，目标 controller 和 broadcaster 为 active；
- 日志无新的 `communication_constraints_violation`、reflex 或 ControlException；
- gateway 显示 `controller_ready=true`，当前仍是 HOLD/未 arm；
- client 使用 `policy_eef_frame=link8`，并成功捕获一次 `F_T_EE`；
- KML-dev server 同时使用 state/action `closed_0_1` override 和正确的 `step_005000.pt`；
- gripper follower 参数为当前标定端点 `0.0/0.8`，开/闭方向做过低风险验证；
- 相机顺序、task 文本和 checkpoint 训练设置一致；
- 工作空间净空、急停/Franka Desk 可用，由现场操作者最后明确 arm。

额外注意：action loss、video loss 收敛不证明真机 contract 正确；先验证 frame、单位、绝对/增量、gripper 方向和相机映射。FastWAM 的 debug video 是模型诊断，不是机械臂 action 对齐正确的证明。

## 8. 相关文档

- `franka_project/markdown/FASTWAM_MIX3_STEP_005000_DEPLOYMENT_AUDIT.md`：该 checkpoint 的完整部署审计和 shadow 对比。
- `franka_project/markdown/FASTWAM_MULTITASK_INTERACTIVE_DEPLOYMENT.md`：当前 server/client 启动方式。
- `franka_project/markdown/ROS2_INTERFACE.md`：ROS 2 topic、pose 与 gripper contract。
- `franka-fci-cpu-isolation.md`：FCI CPU/线程隔离细节。
- `/home/pnp/franka/start_real_validation.sh`、`stop_all.sh`：真机控制栈启动与清理。

如果未来修复改变了上述任何 contract，必须同时更新本文、对应测试和启动指南；不要只改某一个 server/client flag。
