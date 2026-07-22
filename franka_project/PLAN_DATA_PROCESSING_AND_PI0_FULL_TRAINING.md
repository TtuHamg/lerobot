w# Franka 数据处理与 π0 全参数训练计划

> 状态：**COMPLETE：22 条完整轨迹范围已冻结；action15/action30 双版本转换与 QA、π0 3.238B 非 LoRA 全参数 F0 训练及 M0 in-sample 评估均已完成并通过**  
> 版本：v1.5-complete-22of25  
> 日期：2026-07-16  
> 工作目录：`/m2v_intern/tujiahang/Projects/lerobot/franka_project`

## 0. 已确认的执行结论

以下内容已作为首版实验的固定方案；原始 25 条 manifest 仍作为历史 source baseline，D3 及后续实验只使用已冻结的版本化 `22of25` 派生范围。

- [x] 原始 manifest 的历史基线为 25 条严格 `PASS`；可执行范围只包含其中当前实际存在并通过锁定的 22 条 MCAP，缺失的 3 条和全部 32 条 `FAIL` 均不进入首版数据集。
- [x] policy observation 使用双目图像、`current measured EEF` 和当前 gripper；**不输入 qpos**。
- [x] policy action 使用由 `future measured EEF` 和 future measured gripper 构造的 Cartesian waypoint chunk；它们是现有数据条件下的监督代理，不冒充遥操作端或夹爪控制器收到的 desired command。
- [x] 首轮 `obs15/action15` 实验中，22 条执行范围 episode 全部进入 optimizer 的训练池；不按 A/B 划分，也不建立独立 validation/test split。
- [x] qpos 只保留用于基础数据审计和未来底层控制，不作为首版 VLA 输入，也不参与 sampling/checkpoint selection。
- [x] 从同一训练池固定随机抽取一部分 anchor 监控 loss；该指标属于 in-sample `train_monitor_loss`，不是独立验证集上的泛化指标。
- [x] 两个频率版本分别对完整 22 条执行数据计算 LeRobot 风格 statistics，并用对应 model-visible 10D state/7D action 的 mean/std 做 π0 归一化；不进行 unnormalized baseline，也不跨版本混用 stats。
- [x] D3 硬对账目标为每版主 table `8,793` rows、`7,465` 个完整 horizon anchors；model-visible action statistics 的 `count` 分别为 action15 `373,250`（`7,465 × 50`）和 action30 `746,500`（`7,465 × 100`）。
- [x] “全量训练”指从预训练 `lerobot/pi0_base` 初始化后做**全参数微调**：不用 LoRA/PEFT，不冻结视觉编码器，也不只训练 action expert；不从本轮小数据随机初始化 3B 级模型。
- [x] 每条 episode 先裁剪到 cam1、cam2、EEF、gripper 的公共有效时间区间；缺失 cam2 的开头不填零图，也不跨 episode 补齐。
- [x] 同一批 22 条执行数据生成两个独立、可对账的数据版本：`obs15/action15` 与 `obs15/action30`；两个版本的真实相机图像都只保留约 15 Hz，不把图像复制成 30 Hz。
- [x] `obs15/action15` 将 EEF action 对齐到相机时间轴，π0 action chunk 为 50 步、名义 3.33 s；S0/S1/F0/M0 首轮训练和离线实验只使用这一版。
- [x] `obs15/action30` 保留每个相机间隔内的两个约 30 Hz EEF waypoint，action chunk 为 100 步、名义 3.33 s；首轮只完成转换、statistics 和 QA，等 15 Hz baseline 复核后再决定是否训练。
- [x] `obs15/action15` 与 `obs15/action30` 两版数据均已完成转换、statistics 和 D4 QA；F0 按约定只训练 `obs15/action15`，未在 action30 版上启动训练。
- [x] F0 已完成 π0 3,238,048,528 个 `requires_grad` 参数的非 LoRA/PEFT 全参数训练，共 `37,330` steps / 10 effective epochs；固定 monitor loss 从 `0.57267639` 降至 `0.05408542`。
- [x] M0 已达到 `M0_COMPLETE`：覆盖 22 episodes、7,465 anchors 和 3 个推理 seeds；结果属于 teacher-forced、in-sample 离线评估，不代表 held-out 泛化或真机成功率。
- [x] D3 与离线训练以“与采集时为同一台机器，且 configured EE frame/TCP 未改变”为执行假设；该假设不构成真机 rollout 授权。
- [x] 启用 W&B online 记录训练与监控 loss 曲线；设置 `disable_artifact=true`，不向 W&B 上传 model checkpoint。
- [x] 首轮只做数据转换、全参数训练和离线评估；**不连接 Franka，不执行真机 rollout**。
- [x] 唯一任务语言指令固定为：`stack the cups`。

如果上面任一项需要调整，应先修改本文，再继续相应执行阶段。

## 1. 目标、范围与完成定义

### 1.1 目标

1. 将 2026-07-14 采集的 Franka/FR3 MCAP 数据转换为可复现、可审计、互相可对账的 `obs15/action15` 与 `obs15/action30` 两个本地数据集版本。
2. 用 `current measured EEF` 构造 proprioception，用未来实测 EEF 构造 Cartesian action supervision。
3. 从 `lerobot/pi0_base` 开始进行非 LoRA 的全参数训练。
4. 用训练池随机监控子集跟踪 in-sample loss，并在全部训练轨迹上检查离线轨迹重建效果；不将结果解释为未见数据泛化能力。

### 1.2 本轮不包含

- 不修改 `/ytech_milm/collect_data_103/frank3/20260714/DATA` 下的任何原始文件。
- 不使用 `WARN` 或 `FAIL` episode 扩充训练集。
- 不将 qpos、haptic pose/twist/button 输入首版 policy。
- 不声称 future measured EEF 是采集时控制器收到的 desired EEF。
- 不做追帧/长间隔的专项修复；只保留并报告时间统计。
- 不上传 Hugging Face Hub；W&B 仅上传 metrics、run config 和系统信息，不上传 model checkpoint/dataset artifact。
- 不做真机部署、IK/controller 接线或成功率测试。
- 如必须修改 `src/lerobot/` 核心代码，先停止并说明原因；优先把 adapter、processor、sampler 和 evaluator 全部放在 `franka_project/`。

### 1.3 完成定义

本轮完成需要同时交付：

1. 可重复运行的数据审计与转换代码；
2. 两个转换后数据版本、各自的全量 statistics 和数据质量报告；
3. 基于 `obs15/action15` 的 π0 全参数 smoke、过拟合测试和正式训练记录；
4. 按 `train_monitor_loss` 选择的 checkpoint，以及全部训练轨迹上的 in-sample 离线结果；
5. 环境、命令、配置、随机种子、数据和代码版本的完整 manifest；
6. W&B 上可查看训练与 `train_monitor_loss` 曲线，并能从本地 manifest 追溯 project/run ID；
7. 明确说明离线结果的能力边界，并在任何真机运行前再次等待用户确认。

## 2. 已确认的数据基线

### 2.1 数据入口

原始 source baseline 的审计只以如下 manifest 为入口，不递归扫描整个数据目录；D3 的可执行入口由 2.1.1 的派生 manifest 单独冻结：

```text
/ytech_milm/collect_data_103/frank3/20260714/DATA/manifests/train_passed.yaml
```

现有 manifest 记录：

| 项目 | 数值 |
|---|---:|
| 严格 PASS episode | 25 |
| FAIL episode | 32 |
| PASS 总时长 | 698.374 s |
| cam1 近似总帧数 | 10,308 |
| 图像分辨率 | 480 × 640 × 3 |
| cam1/cam2 名义频率 | 约 15 Hz |
| current EEF 名义频率 | 约 30 Hz |
| Franka joint state | 7 DoF，约 1 kHz |

参考加载器：

```text
/ytech_milm/collect_data_103/frank3/20260714/DATA/tools/frank3_dataset.py
```

它目前返回：

```text
images.cam1 : uint8 [T,H,W,3]
images.cam2 : uint8 [T,H,W,3]
qpos       : float64 [T,7]
gripper    : float64 [T,1]
ee_pose    : float64 [T,7] = xyz + quaternion(x,y,z,w)
timestamps : int64 [T]，cam1 log_time，单位 ns
```

参考说明：[实际采集数据记录&ReadMe.pdf](./实际采集数据记录&ReadMe.pdf)。

#### 2.1.1 版本化 `22of25` 执行范围

上面的 25 条记录是原始 manifest 的历史事实，不改写为“25/25 可转换”。D0 实际确认只有 22 条声明的 MCAP 文件存在，因此后续 D3、D4、S0、S1、F0 和 M0 统一以如下派生 manifest 为唯一可执行入口：

```text
/m2v_intern/tujiahang/Projects/lerobot/franka_project/manifests/frank3_train_passed_22of25_v1.yaml
```

冻结范围如下：

| 项目 | 数值 |
|---|---:|
| source manifest episode | 25（历史基线） |
| D3 可执行 episode | 22 |
| 显式排除的缺失 MCAP | 3 |
| 可执行总时长 | 608.013 s |
| raw cam1 / cam2 message | 8,987 / 9,007 |
| 每版预期主 table rows | 8,793 |
| 两版共同完整 horizon anchors | 7,465 |
| action15 model-visible stats `count` | 373,250 = 7,465 × 50 |
| action30 model-visible stats `count` | 746,500 = 7,465 × 100 |

供转换器和 QA 直接断言的无格式整数为：

```yaml
expected_main_rows_per_dataset: 8793
valid_full_horizon_anchors: 7465
pi0_action15_stats_count: 373250
pi0_action30_stats_count: 746500
```

显式排除项为：

```text
run_20260714_202237_00032
run_20260714_175543_00008
run_20260714_165338_00005
```

范围 provenance：

```yaml
source_manifest_sha256: d2ba4dc819eafd0163ed6673f53ea6e6cc7d492b075978c4db925c5b68dcf38b
scope_config_sha256: 2dd95dcd5b9b953abd816d7863af1b38d92275923af884eadcbd07527273bd56
derived_manifest_sha256: 76b839b29840714dc6a0c58261d2709b25dfd846fe7f13761787fd6d02b17a77
scope_content_sha256: 86402a4138002b1ff6e02d95e7f434c5ef3f658c362fb88253ff8b1c44a69461
```

22 条 MCAP 均按 byte size + streamed full-file SHA-256 锁定。任一文件 size/hash 或范围 config 改变时，当前 scope lock 失效，必须重新冻结并重跑相应审计；不得临时将缺失项、`viz_cache` 或其他 episode 混入当前版本。

### 2.2 EEF 与 gripper 字段语义

EEF 来源为：

```text
/franka_robot_state_broadcaster/current_pose
```

因此它表示机器人已经达到的 **measured current EEF pose**。现有数据没有 `last_desired_pose` 或等价的控制器 desired EEF topic。

当前 pose 被解释为 robot base 中 configured end-effector 的 `O_T_EE`。bag 中没有足够的数值配置可独立复原并证明采集时的 `F_T_EE`/`NE_T_EE` 或自定义 TCP，因此本轮 D3 和离线训练明确采用以下执行假设：运行机器与采集机器相同，configured EE frame/TCP 也保持不变。该假设允许本轮数据语义冻结与离线实验继续，但**不单独授权未来真机 rollout**；部署前仍需再次核对 EE/TCP 并由用户单独批准。

首版数据集必须写入以下 provenance，避免以后误用：

```yaml
observation_pose_source: measured_current_pose
action_source: future_measured_current_pose
action_is_expert_command: false
action_semantics: realized_successful_cartesian_waypoint_proxy
gripper_observation_source: measured_gripper_joint_state
gripper_action_source: future_measured_gripper_joint_state
gripper_action_is_expert_command: false
qpos_is_policy_input: false
```

这个选择的含义是：模型学习成功演示中“实际走出来的 Cartesian 轨迹和夹爪轨迹”，其中可能已经包含底层控制器跟踪滞后；它不是对原始遥操作或夹爪 desired command 的精确还原。

### 2.3 A/B 构型的处理边界

已观察到原始 25 条 PASS 记录中存在两簇 qpos 构型，数量约为 A=5、B=20；这是原始数据基线中的历史观察。当前 `22of25` 执行范围仍以快速验证 task 为目标，且 **不围绕 A/B 做额外处理**：

- 不生成 A/B train/validation/test split；
- 不做 A/B balance sampler；
- 不用 A/B 指标选择 checkpoint；
- 不把 A/B 或 qpos 输入模型。

qpos 仍原样保存到 audit sidecar，便于检查 joint 顺序、范围和未来底层 controller 接口，但 A/B 聚类不再是数据转换或训练的硬门槛。

## 3. 共同数据契约与两个频率版本

### 3.1 时间轴和同步

两个版本共享同一套 15 Hz observation 时间轴，并以 MCAP `log_time_ns` 对齐，不按消息序号对齐：

1. 以 cam1 frame 顺序作为主时间轴，名义 observation `fps=15`；
2. 每条 episode 的起止先裁剪到 cam1、cam2、EEF、gripper 都有数据覆盖的公共有效时间区间；
3. 对保留下来的每个 cam1 时间 `c_i`，cam2、EEF、qpos、gripper 使用“时间戳最近且不晚于 `c_i`”的因果 zero-order hold；
4. 不使用参考 loader 在 cam2 尚未开始时的全零图像 fallback；任何必需流不存在或超过对齐时效阈值时，该 frame 标为无效，不伪造 observation；
5. D0 会报告 cam2/EEF/gripper 的 alignment age 分布，并在 resolved data config 中冻结各流阈值；默认上限以各流名义周期加 jitter 为依据，而不是根据训练结果调整；
6. 保存 cam1 原始 `log_time_ns`、相邻真实 `dt`、每个被选 source message 的时间戳和 alignment age 到审计 sidecar；
7. 不跨 episode 对齐，不对图像或 EEF pose 做数值插值；
8. 不额外重采样来专项修复追帧，长间隔只进入报告，不单独成为阻断条件。

公共裁剪和有效性过滤后，记某条 episode 共有 `M` 个按时间递增的 camera observation anchors：

```text
c_0, c_1, ..., c_(M-1)       # cam1 主时间戳，名义 15 Hz
```

cam1 使用自身 frame；cam2 和 current EEF/gripper 分别携带独立 source timestamp。这样既不把 30 Hz EEF 与第几个 camera frame 直接按序号配对，也不把 15 Hz 图像上采样成 30 Hz。

这是有意接受的首版近似：相机帧索引按 15 Hz 解释，而少数真实 interval 可能不是严格的 66.7 ms。后续如要专项处理追帧，应生成新的 dataset version，而不是覆盖这两个版本。

### 3.2 两个数据版本总览

| 数据版本 | observation | action waypoint | chunk | 名义 horizon | 首轮用途 |
|---|---:|---:|---:|---:|---|
| `franka_current_eef_obs15_act15_v1` | 15 Hz 双目 + state | 15 Hz | 50 | 3.33 s | 转换、QA、S0/S1/F0/M0 |
| `franka_current_eef_obs15_act30_v1` | 15 Hz 双目 + state | 30 Hz | 100 | 3.33 s | 转换、statistics、QA；暂不训练 |

两版共享相同的 22 条 `22of25` 执行 episode、公共裁剪规则、camera anchor ID、state 表示、语言指令和几何 convention。30 Hz 版不是把同一图像复制两次形成 30 Hz observation；它使用独立 action stream 和项目本地 sampler。

### 3.3 图像输入

```text
observation.images.cam1 : RGB uint8 [H,W,3]
observation.images.cam2 : RGB uint8 [H,W,3]
```

- 两个相机都输入 π0；
- 转换阶段不改变原始宽高比，不自行做颜色归一化；
- resize/pad/normalization 交给 π0 官方 processor；
- 相机 topic 中没有可用的 CameraInfo 标定消息，因此首版不做三维重投影或基于标定的视觉增强；
- QA 中生成同步双目 contact sheet，人工抽查 episode 的开始、中间、结束和抓放事件。

### 3.4 Observation state

模型看到的 proprioception 为：

```text
observation.state = [x, y, z, rot6d_0..5, gripper]  # 10D
```

定义：

- `x,y,z`：`current_pose` 在消息原始基坐标系下的位置，单位 m；
- `rot6d`：旋转矩阵前两列的 6D representation，列优先展开；
- `gripper`：当前夹爪开合量，先核对原始正负方向和物理范围，再用固定映射规范到 `[0,1]`；
- 原始 quaternion 为 `xyzw`，转换前归一化，并通过相邻点 dot product 检查/修复 `q` 与 `-q` 的符号跳变；
- 执行前必须确认 `current_pose.header.frame_id` 在全部 episode 中一致，并记录 EEF 是哪一个实际 frame/TCP。若不一致，停止转换。

不用 qpos 替代 EEF，也不把 qpos 拼到 `observation.state`。

### 3.5 Action supervision

记 camera anchor `c_t` 对齐到的 current measured pose/gripper 为 `p_t, R_t, g_t`。两个版本只在 future target 时间网格上不同；对任意版本的第 `j` 个 target，先从 MCAP 选择其 future measured pose/gripper `p*_(t,j), R*_(t,j), g*_(t,j)`，再统一编码为：

```text
delta_position(t,j) = p*_(t,j) - p_t                 # 3D，base frame
delta_rotation(t,j) = Log( R_t^T · R*_(t,j) )        # 3D，current-EEF/body frame，rad
gripper_waypoint(t,j) = g*_(t,j)                     # 1D，absolute measured proxy

action(t,j) = [delta_position, delta_rotation, gripper_waypoint]  # 7D
```

#### 3.5.1 `obs15/action15`

对 `k=1...50`：

```text
target_time_15(t,k) = c_(t+k)
```

EEF/gripper 使用不晚于该 future camera timestamp 的最新 measurement。参数为：

```text
K15 = 50 waypoints
nominal action rate = 15 Hz
nominal first lead = 1 / 15 s
nominal horizon = 50 / 15 = 3.33 s
```

#### 3.5.2 `obs15/action30`

为保持和 15 Hz 版相同的 camera anchors 与物理 horizon，每个相邻 camera interval 放置两个 action target。对 `k=1...50`，1-based waypoint 编号定义为：

```text
target_time_30(t,2k-1) = midpoint(c_(t+k-1), c_(t+k))
target_time_30(t,2k)   = c_(t+k)
```

每个 target time 同样选择“不晚于该 target 的最新”EEF/gripper measurement，不对 pose 插值。参数为：

```text
K30 = 100 waypoints
nominal action rate = 30 Hz
nominal first lead = 1 / 30 s
nominal horizon = 100 / 30 = 3.33 s
```

这个构造提供一个强制对账关系：在使用相同 source-message selector 时，30 Hz chunk 的每个偶数 waypoint 应与 15 Hz chunk 对应 waypoint 使用同一个绝对 EEF/gripper target，即：

```text
absolute_action_30(t, 2k) == absolute_action_15(t, k),  k=1...50
```

若某个 target 只能匹配到过旧或缺失的 EEF/gripper measurement，该 action slot/anchor 无效并进入 QA 报告；不使用超时旧消息或 episode 尾部 padding 来伪造有效 30 Hz 标签。机器人静止时不同有效 source messages 数值相同则属于真实数据，不视为伪造。

#### 3.5.3 两版共同的几何与执行语义

chunk 内每个 waypoint 都相对同一个 current EEF anchor `t`，不是把逐步 delta 直接累加。解码为：

```text
p_target(t,j) = p_t + delta_position(t,j)
R_target(t,j) = R_t · Exp(delta_rotation(t,j))
```

这是一个刻意冻结的混合约定：平移 delta 用 robot base frame，旋转 delta 用 current EEF/body frame。它不是把完整 `T(t)^-1 · T(t+k)` 的 6D twist 直接送入模型。训练、离线解码和未来 controller adapter 必须读取同一份 convention config，不允许一侧改成 spatial/left-multiply rotation。

这样 action 不直接依赖 qpos，但未来真机执行仍需要底层 controller 根据实时 qpos 做 IK/differential IK/Cartesian control 和安全约束。

`chunk_size=50/100` 只定义相应版本的模型预测 horizon，不代表未来真机会连续开环执行整个 chunk。30 Hz 数据标签也不自动保证真机 30 Hz 控制；未来仍需 controller adapter、action queue、时钟和安全评审。`n_action_steps` 属于后续真机闭环方案，本轮不冻结。

### 3.6 落盘 action 与取样约定

#### 3.6.1 标准 `obs15/action15` 版本

为消除 π0 默认 `action_delta_indices=[0,...,49]` 与本文 `t+1...t+50` 之间的 off-by-one，v1 明确采用以下 carrier schema：

```text
公共裁剪后有 M 个 camera anchors，索引 0...M-1
LeRobot 写入 N = M-1 行 observation，索引 i=0...M-2

on-disk action[i] = action_15_carrier[i]
                  = absolute target selected at c_(i+1)
                  = [xyz(3), quaternion_xyzw(4), gripper_0_1(1)]  # 8D
```

因此对 LeRobot anchor `t` 查询 on-disk carrier rows `t...t+49`，恰好得到 camera targets `c_(t+1)...c_(t+50)`。最后一个 camera anchor 不作为 policy observation row，但保留在 audit mapping 中并可作为 future target；不会制造重复终点 sentinel。

项目本地 dataset adapter 在 `__getitem__` 后执行：

```text
on-disk absolute action chunk [50,8]
    + current observation.state [10]
    -> model-visible anchor-relative action chunk [50,7]
```

adapter 同时向 policy 暴露有效 feature schema `action.shape=[7]`。raw 8D carrier 只是可审计的落盘表示，不直接进入 π0，也不使用其默认 action statistics。

#### 3.6.2 双频 `obs15/action30` 版本

LeRobot 当前用单一 dataset `fps` 把 `action_delta_indices` 转成 row offsets，不能仅靠标准 row sampler 表达“observation 15 Hz、action 30 Hz”。因此该版本采用项目内的显式 dual-rate extension：

```text
observation/video table : N = M-1 rows，fps=15，不重复 camera frame
actions_30hz table      : 2*(M-1) rows，每个 camera interval 两个 8D absolute carriers
anchor map              : camera anchor -> 100 个 future action row IDs + target/source timestamps
metadata                : observation_fps=15, action_fps=30,
                          requires_project_dual_rate_adapter=true
```

对 interval `i -> i+1`：

```text
actions_30hz[2*i]   = absolute target selected at midpoint(c_i, c_(i+1))
actions_30hz[2*i+1] = absolute target selected at c_(i+1)
```

项目 sampler 对 camera anchor `t` 读取 `actions_30hz[2*t : 2*t+100]`，再结合当前 10D state 转换为 model-visible `[100,7]` action。该版本不得通过 stock `lerobot-train`/`resolve_delta_timestamps` 静默加载；若未启用 dual-rate adapter，入口必须 fail fast。

π0 本地实现的 `chunk_size` 是可配置的，100-step chunk 不改变 state/action projection 的参数 shape，但会加长 action-token 序列、增加显存和计算量。30 Hz 训练开始前仍必须单独做 checkpoint-load、forward/backward、显存和吞吐 smoke；本轮先不训练它。

### 3.7 为什么需要项目本地 Cartesian adapter/processor

LeRobot 现有通用 relative-action processor 主要对 action 和 state 的相同维度做数值减法。它不能正确完成 quaternion/rotation 的群运算，也不能直接表达上述 anchor-relative future waypoints。

因此会在 `franka_project/` 内实现几何明确的 dataset adapter、sampler 和 processor：

- 从对应版本的 absolute carrier stream 取得 50/100 个 future targets；
- 计算 7D anchor-relative chunk；
- 使用全部 22 条执行范围训练数据上、完成几何变换之后的 action statistics 做归一化；
- 显式调用 `decode(predicted_delta, anchor_state)` 将预测解码回 base-frame Cartesian target；
- 用 SO(3) round-trip、已知旋转、batch/chunk shape 和 checkpoint reload 单元测试验证。

不会启用 stock `policy.use_relative_actions` 来处理 EEF。

postprocess 不得依赖一个全局或可变的“last state”缓存。离线批量评估必须显式携带与每个 prediction 一一对应的 `anchor_state/anchor_id`；乱序 batch、batch size >1、并发调用和 checkpoint reload 都要测试，防止把一个样本的 delta 解码到另一个样本的 current EEF 上。

### 3.8 Episode 尾部和 padding

现有 π0 虽会构造 action padding 信息，但当前 loss 没有可靠地按时间维忽略 padded action。首版采用更简单且可验证的方案：

```text
15 Hz: 只采样能够提供完整 50-step / t+1...t+50 camera targets 的 anchor
30 Hz: 只采样能够提供完整 100-step / 同一 t...t+50 camera interval 的 anchor

N = M - 1                             # 15 Hz observation rows
valid_anchors(episode, both versions) = max(0, M - 50)
```

两版的有效 anchor ID 因而可以一一对账；任何 alignment-invalid slot 会进一步使包含它的 chunk 无效。不会把“重复最后一帧/EEF”作为真实训练标签。未来若要利用尾部 anchor，必须先实现并测试 masked loss，作为单独版本变更。

### 3.9 qpos 与审计 sidecar

qpos 不放入 policy-visible feature。它按原始 episode/frame 保存到独立 audit sidecar，内容至少包括：

- 7 个关节角和 joint names；
- raw timestamp / aligned frame index；
- joint limit、速度跳变和缺失值检查；
- 与 EEF/frame 对账所需的索引。

这样既保留未来底层 controller 和数据审计所需信息，又能降低训练配置误把 qpos 当成 observation 的风险。v1 不生成或消费 A/B 标签。

### 3.10 Task instruction

π0 需要语言 task。虽然 bag/manifest 没有 task 字符串，本批数据的 instruction 已由采集者固定为：

```yaml
task_instruction: "stack the cups"
task_description_zh: "把杯子叠起来"
```

全部 22 条执行范围 episode 使用同一 instruction 和同一个 `task_index`。转换 QA 必须逐条核对该映射，不能自动改写大小写或追加另一种任务表述；π0 tokenizer 所需的结尾换行由官方 processor 处理。

## 4. 全量训练数据、statistics 与监控子集

### 4.1 全部 22 条执行 episode 用于训练

首轮训练只有一个数据用途：

```text
train dataset = franka_current_eef_obs15_act15_v1
train = 版本化 22of25 范围内的全部 22 条严格 PASS episode
validation = none
test = none
```

`obs15/action15` 中所有能够提供完整 50-step action horizon 的 anchor 都进入训练采样池。不按 A/B、episode 或 frame 建立 held-out split，也不进行 A/B 平衡采样。

`franka_current_eef_obs15_act30_v1` 使用相同 22 条 episode 和 camera anchors，但本轮只做转换、statistics、对账和 QA，不加入 S0/S1/F0/M0。必须先提交 15 Hz baseline 结果，再单独确认 30 Hz 训练配置和成本。

该方案服务于当前目标——快速确认采集 task、数据格式和 π0 全参数训练链路能否拟合。由于没有未见数据，本轮不能估计泛化性能。

### 4.2 标准 LeRobot statistics

必须参考以下已有数据集的格式生成 statistics：

```text
/ytech_milm_intern/tujiahang/.cache/huggingface/lerobot/
tuuy/lerobot_so_arm101_task0_new/meta/stats.json
```

转换时使用 LeRobot 的 episode stats/aggregation 逻辑，为两个数据版本各自主 observation table 中的落盘 features 生成标准：

```text
meta/stats.json
```

每个数值 feature 至少包含：

```text
min, max, mean, std, count, q01, q10, q50, q90, q99
```

图像按 LeRobot 约定保存 `[0,1]` 范围的逐通道 statistics。标准文件覆盖对应版本全部 22 条执行 episode 的 8,793 个 converted observation rows，而不是某个子集。

30 Hz 的独立 action table 不是 stock LeRobot feature，不能冒充被标准 `meta/stats.json` 覆盖；它另生成相同字段结构的：

```text
meta/action_30hz_carrier_stats.json   # 8D absolute carrier，仅审计
```

该文件不直接用于 π0 normalization。metadata 必须明确区分 observation-table stats、30 Hz carrier stats 与下节的 model-visible effective stats。

### 4.3 π0 实际使用的 effective statistics

on-disk action carrier 是 8D absolute measured pose/gripper，但模型看到的是几何变换后的 7D anchor-relative action，因此不能直接把 carrier stats 用于 π0 action normalization。

除标准 `meta/stats.json` 外，必须生成同样字段结构的：

```text
meta/pi0_eef_stats.json
```

其中 π0 消费的核心 keys 为：

```text
observation.state : 10D，统计对应版本全部 22 条的 8,793 个 converted observation rows
action            : 7D，15 Hz 版统计 7,465 个有效 anchor 的 50 个 transformed waypoints；count=373,250
                         30 Hz 版统计 7,465 个有效 anchor 的 100 个 transformed waypoints；count=746,500
```

每个数据版本在自己的 `meta/pi0_eef_stats.json` 中记录 `observation_fps`、`action_fps`、`chunk_size` 和 source dataset hash。action statistics 必须在执行 absolute pose → anchor-relative Cartesian action 后计算，action `count` 对应该版本所有有效 anchor-horizon waypoint 数量。训练入口显式注入与 dataset version 匹配的文件，不得回退到 on-disk 8D carrier stats，也不得在 15/30 Hz 版本之间混用。

π0 normalization mapping 固定为：

```yaml
VISUAL: IDENTITY
STATE: MEAN_STD
ACTION: MEAN_STD
```

这里的 `VISUAL: IDENTITY` 只表示不使用 dataset image mean/std 做第二次标准化；图像仍经过 π0 官方的缩放、resize/pad 等处理。state 和 action 必须使用上述 mean/std 做 normalize/unnormalize，不执行 unnormalized 对照实验。

启动时断言：

- source manifest、冻结的 22of25 derived manifest/scope lock、rate/chunk config、geometry transform config 和该版本全部 required stats 文件的 SHA256 一致；
- state/action stats shape 分别为 10/7；
- 所有统计值有限，`count` 与转换索引对账；
- 零或极小 std 使用官方 epsilon 保护并单独告警；
- normalize → unnormalize round-trip 在数值容差内恢复原值；
- checkpoint reload 后 stats 与 processor 完整恢复。

### 4.4 训练池随机监控子集

从 `obs15/action15` 的全部有效训练 anchors 中一次性随机抽取固定子集：

```text
name: train_monitor_subset
default size: min(1024, ceil(10% * valid_train_anchors))
sampling seed: 固定并写入配置
```

anchor ID 列表、seed 和 SHA256 在训练前冻结。每个 checkpoint 在 `model.eval()`、`no_grad()` 和固定 flow-noise/prediction seeds 下计算：

```text
train_monitor_loss
```

该子集仍属于训练池，相关 anchor 仍可被 optimizer 采样；它不是 held-out validation/test。`train_monitor_loss` 只用于观察拟合趋势、发现发散并选择训练过程中的 checkpoint，不能用于声称未见数据上的泛化能力。固定子集而非每次重新抽样，是为了让不同 checkpoint 的曲线可比较。

### 4.5 W&B 记录契约

W&B 为正式实验的必需记录通道，首版固定为 online metrics logging：

```yaml
wandb:
  enable: true
  project: franka_pi0_full_eef
  entity: null          # 使用当前已认证账号的默认 entity
  mode: online
  disable_artifact: true
  add_tags: true
```

至少记录以下曲线，且全部使用同一 `global_step`：

```text
train/loss
train/lr
train/grad_norm
train/samples_per_s
monitor/train_monitor_loss
monitor/translation_ade_mm
monitor/rotation_ade_deg
monitor/gripper_mae
```

正式 run 默认每 50 个 optimizer steps 写一次 train metrics；`train_monitor_loss` 在每个 epoch/checkpoint 计算并写入。若实测 W&B I/O 明显影响吞吐，只能在保留足够曲线分辨率的前提下调整频率，并把变更写入 resolved config。

只有 rank 0 初始化和写入 W&B。每个 local `run_id` 对应唯一 W&B run；resume 必须复用原 W&B run ID，不能产生一条新的断裂曲线。W&B project、entity、run ID 和 URL 写入本地 `run_manifest.json`。

checkpoint 仍按计划保存在 `franka_project/experiments/`，用于 resume 和 checkpoint selection。必须同时满足：

- `wandb.disable_artifact=true`，使 LeRobot 的 `WandBLogger.log_policy()` 直接跳过；
- 不调用自定义 `wandb.Artifact`、`wandb.save` 或 `log_model`；
- `policy.push_to_hub=false`；
- 不上传 dataset、model weights 或 checkpoint directory。

run config、标量 metrics 和 W&B 自动采集的系统指标可以同步。若 online 鉴权或网络不可用，不得静默关闭 W&B；先停止并报告，是否改为 offline mode 后续再确认。

## 5. 项目目录规划

确认计划后创建以下结构；当前阶段只新增本文：

```text
franka_project/
├── PLAN_DATA_PROCESSING_AND_PI0_FULL_TRAINING.md
├── 实际采集数据记录&ReadMe.pdf
├── configs/
│   ├── data/
│   ├── train/
│   └── monitor/
├── src/franka_eef_pipeline/
│   ├── mcap_reader.py
│   ├── geometry.py
│   ├── action_chunk.py
│   ├── dual_rate_dataset.py
│   ├── processors.py
│   ├── sampler.py
│   └── metrics.py
├── scripts/
│   ├── audit_raw_data.py
│   ├── convert_to_lerobot.py
│   ├── build_pi0_stats.py
│   ├── build_train_monitor_subset.py
│   ├── verify_lerobot_dataset.py
│   ├── train_pi0_full.py
│   └── eval_pi0_train_monitor.py
├── tests/
├── data/lerobot/
│   ├── franka_current_eef_obs15_act15_v1/
│   └── franka_current_eef_obs15_act30_v1/
├── artifacts/
│   ├── raw_audit/
│   ├── stats/
│   ├── train_monitor/
│   └── conversion/
├── experiments/pi0_full_eef_obs15_act15_v1/<run_id>/
├── reports/
└── .cache/
```

每个正式 run 至少包含：

```text
resolved_config.yaml
run_manifest.json
train.log
environment/
wandb/
checkpoints/{best_train_monitor,last,milestone}/
in_sample_eval/
```

所有生成数据和 checkpoint 默认本地保存且不进 Git；实现时只添加 project-local ignore 规则，不覆盖用户现有 `.gitignore` 修改。

## 6. 分阶段执行计划与硬门槛

### Phase E0：环境与版本基线

任务：

- 在项目目录内建立可复现运行环境，优先使用仓库 `uv.lock`；当前 shell 未发现 `uv`，先解决本地工具入口，不修改全局环境；
- 确认 MCAP/ROS2 解码、ffmpeg、PyTorch、CUDA、LeRobot 和 π0 checkpoint 可用；
- 确认 W&B Python 依赖、online 鉴权和网络连通；只检查登录状态，不把 API key 写入日志或 manifest；
- 将 Hugging Face/Torch/uv cache 指向 `franka_project/.cache/`；
- 记录 Git commit、dirty diff hash、`uv.lock` hash、Python/PyTorch/CUDA/cuDNN、GPU 和磁盘信息；
- 当前只读盘点显示有 2 张 NVIDIA A800-SXM4-80GB；实际运行前重新核对占用。

通过条件：一个 PASS episode 能通过官方 loader 完整解码，环境 manifest 可重建，测试 W&B run 能写入一个标量并正常结束且不产生 artifact。

停止条件：依赖无法锁定、模型权重不完整、数据解码需要改写原始文件、W&B online 无法鉴权或连接。

### Phase D0：原始数据 inventory 与完整性审计

任务：

- 校验 manifest SHA256、25 条路径和 MCAP 文件存在性；
- 复算 episode 数、时长、帧数、topic、shape、dtype、timestamp 单调性；
- 检查 EEF quaternion norm、NaN/Inf、gripper 范围、joint names/order；
- 统计每条 episode 的真实 frame `dt`、最大 gap、各流起止覆盖、公共有效区间、cam2/EEF/gripper 对齐偏差；
- 对比 MCAP log time 与 ROS header stamp 的可用性，冻结本轮使用 `log_time_ns` 的证据和 source timestamp 审计字段；
- 确认 `current_pose` frame_id/TCP 语义；
- 生成机器可读 JSON/Parquet 和人可读 Markdown 报告。

通过条件：25 条严格 PASS 均可复现读取，关键字段一致，所有异常都有 episode/frame 定位。

停止条件：数量或 manifest 与当前基线不符、关键 topic/field 不一致、EEF frame 不一致、存在未解释的非有限值。

### Phase D1：任务、gripper 与表示配置审计

任务：

- 保留原始 25 条 source manifest 的 D0 历史结论，同时核对派生范围全部 22 条都是同一 `stack the cups` task，并写入唯一 `task_index`；
- 确认 gripper 开/闭方向和 `[0,1]` 映射；
- 冻结 10D state、8D carrier、7D action、坐标系和 quaternion convention；
- 冻结公共区间裁剪、alignment-age 阈值、15 Hz target selector 和 30 Hz midpoint/endpoint target selector；
- 冻结 `obs15/action15: K=50` 与 `obs15/action30: K=100` 的双版本 schema 和交叉对账规则；
- 冻结全量 statistics schema、normalization mapping 和 hash 规则；
- 冻结 `train_monitor_subset` 的抽样规则与 seed；
- qpos 只做 joint name/order、范围、缺失值审计，不做 A/B 聚类或划分。

通过条件：task/gripper/几何表示/stats/monitor 配置全部冻结；两版使用同一个已锁定的 22 条 derived scope，且首轮 optimizer 训练用途只指向 `obs15/action15`。

停止条件：任一 episode 不是该任务、gripper 映射无法确定、EEF/frame convention 不一致。

### Phase D2：Cartesian action pipeline spike

先用至少 2 条不同 episode，只实现内存/临时小样本，不立刻转换全量。

任务：

- 实现 quaternion → rotation matrix → rot6d 和 SO(3) Log/Exp；
- 实现 15 Hz `t+1...t+50` 的 50-step anchor-relative 7D action chunk；
- 实现 30 Hz midpoint/endpoint 的 100-step anchor-relative 7D action chunk 和独立 action table/anchor map；
- 验证 15 Hz on-disk 8D rows `t...t+49` 与 raw camera targets `t+1...t+50` 精确对应；
- 验证 `action30(t,2k)` 与 `action15(t,k)` 的 absolute target/source timestamp 一致；
- 实现两个版本各自的 full-horizon anchor sampler；
- 实现两个版本的 effective state/action stats 计算及 normalization，并验证不会读取默认 8D carrier stats 或混用 15/30 Hz stats；
- 验证 custom processor 前后 feature shape；
- 验证 action encode/decode round-trip；
- 验证 checkpoint 保存/加载后 processor、geometry convention 和 statistics 不丢失；
- 验证显式 anchor 的乱序 batch、batch size >1 和并发 decode 不串样本。

数值门槛：

- quaternion/rotation round-trip geodesic error `< 1e-5 rad`；
- position encode/decode max error `< 1e-6 m`；
- 无 NaN/Inf；
- 15 Hz chunk 的第一帧确为 `t+1`，最后一帧确为 `t+50`；
- 30 Hz chunk 恰有 100 个 future slots，偶数 slots 与 15 Hz targets 对账，最后一帧同样落在 `c_(t+50)`；
- 不产生跨 episode target 或 padded target；
- normalize → unnormalize round-trip 通过，零/极小 std 有 epsilon 保护；
- stats schema 包含 `min/max/mean/std/count/q01/q10/q50/q90/q99`，且无 NaN/Inf。

同时分别统计两个版本所有有效 chunks 的相对旋转角。若存在 `theta >= pi - 0.1 rad`，或观察到 SO(3) Log 在 π 分支附近不连续，则 D2 不通过：先缩短 horizon 或修订 rotation representation，并更新本文。

停止条件：只能通过 stock 逐维相减实现、processor 无法随 checkpoint 恢复、必须静默修改 LeRobot 核心代码。

### Phase D3：全量转换两个数据版本

任务：

- 只按 `frank3_train_passed_22of25_v1.yaml` 的锁定顺序转换全部 22 条 PASS；原始 25 条 manifest 不作为 D3 可执行入口；
- 先按公共有效区间和 alignment validity 生成共享的 `M` 个 camera anchors，绝不以全零 cam2 补齐；
- 写入 `franka_current_eef_obs15_act15_v1`：每条 episode 有 `M-1` 条 15 Hz policy rows、两个 15 Hz camera video、10D observation state 和 8D next-camera-target carrier；
- 写入 `franka_current_eef_obs15_act30_v1`：相同的 15 Hz observation/video rows，加 `2*(M-1)` 条 30 Hz absolute action carriers 和显式 anchor map；
- qpos/raw timestamp/对齐偏差写 audit sidecar，不暴露给 policy；
- 写 task、provenance、raw episode ID 与 frame mapping；
- 为两个版本主 table 的落盘 features 生成标准 `meta/stats.json`，并为 30 Hz sidecar 生成 `meta/action_30hz_carrier_stats.json`；
- 对两个版本全部有效 anchors 分别生成 model-visible `meta/pi0_eef_stats.json`；
- 对账并冻结预期数量：每版主 table `8,793` rows、共同完整 horizon anchors `7,465`、action15 stats `count=373,250`、action30 stats `count=746,500`；
- 生成并冻结 `train_monitor_subset` anchor 列表、seed 和 SHA256；
- 保留原始到 LeRobot episode/frame 的双向索引；
- 产物使用新目录原子写入，不覆盖已有 dataset version。

通过条件：15 Hz observation 数量按 `sum(M-1)=8,793`、30 Hz carrier 数量按 `sum(2*(M-1))=17,586` 精确对账；两版的完整 horizon anchors 均为 `7,465`，model-visible action stats count 分别严格为 `373,250/746,500`；两版 camera anchor ID 相同，30 Hz 偶数 target 与 15 Hz target 全量一致，最后 anchor 的 target/audit mapping 可追溯，随机抽样和全部 schema 检查通过。

停止条件：任意 episode 丢帧无法解释、video/state/action 索引不一致、stats count/shape/hash 不匹配。

### Phase D4：两个 dataset 版本 QA

自动检查：

- 两版 dataset metadata、`observation_fps/action_fps/chunk_size`、feature schema、episode boundaries；
- 随机/边界 frame 解码；
- 两相机、EEF、gripper 的时间同步；
- 公共有效区间内没有零图 fallback，必需流的 alignment age 均在冻结阈值内；
- 15 Hz 50-step 与 30 Hz 100-step action chunk 的几何、horizon 和交叉对账；
- 30 Hz 版未启用 dual-rate adapter 时必须 fail fast，不能被 stock row offsets 误读为 15 Hz action；
- 全部 22 条执行范围 episode 均属于 train，且没有其他 split；
- 每个版本全部 required stats 文件的字段、shape、count、rate metadata、hash 和 processor 注入路径；
- 归一化后 state/action 的有限性、均值、标准差和 outlier；
- `train_monitor_subset` 只引用合法 full-horizon anchors，列表固定可复现。

人工检查：每条 episode 至少检查 start/middle/end，另检查全部 gripper transition 附近窗口。

通过条件：测试全绿，QA 报告签字后才允许加载 π0。

### Stage S0：π0 全参数 smoke（仅 `obs15/action15`）

数据：至少 2 条不同 episode，2–5 optimizer steps。

检查：

- image/state/action shape 正确；
- forward/backward 无 NaN/Inf；
- 顶层 `TrainPipelineConfig.peft=null`；
- `use_peft=false`，不存在 LoRA adapter；
- `freeze_vision_encoder=false`；
- `train_expert_only=false`；
- 启动时记录并断言 `trainable_parameters == total_parameters`；
- vision encoder、VLM、action expert、state/action projection 均有梯度；
- 实际注入的是 10D/7D `pi0_eef_stats.json`，STATE/ACTION 均完成 MEAN_STD normalization；
- normalized state/action、loss 和梯度均无 NaN/Inf；
- normalize/unnormalize 及 processor reload 后结果一致；
- checkpoint save/reload、processor reload 和 resume 正常；
- W&B 能收到 `train/loss` smoke 点，且 run page 中没有 model artifact；
- resume 使用同一 W&B run ID 追加 step，不创建第二条 run；
- 记录单卡 A800 的峰值显存和 step time。

S0 先单 GPU、BF16、batch size 1、gradient checkpointing、关闭 compile。

停止条件：任何预期参数被冻结、梯度覆盖不符、OOM/NaN 无法在保持全参数训练的条件下解决。

### Stage S1：小样本过拟合（仅 `obs15/action15`）

执行状态：**PASS**。300-step 小样本训练、固定 4-sample train-internal monitor、
base→step300 Cartesian error 对比和 model/saved-processors reload 均已完成；详细数值见
`reports/S1_PI0_FULL_OVERFIT_22OF25.md`。本次 S1 为单次 `0→300` invocation，
`resume=false`；通用 resume 合同由 S0 的 step `2→3` 实测链路覆盖。

数据：固定 2 条不同 episode；约 100–300 steps。

通过条件：

- training flow-matching loss 明显下降；
- 同批样本的 Cartesian translation/rotation/gripper error 同步下降；
- reload 前后固定 seed 预测在数值容差内一致；
- resume 后 optimizer、scheduler 和 step 正确延续。

若 S1 不能过拟合，不得通过增加正式训练步数掩盖问题；先检查标签、归一化、processor 和 loss。

### Stage F0：正式全参数 baseline（仅 `obs15/action15`）

执行状态：**PASS / COMPLETE**。双卡 2-step preflight 通过后，正式训练从同一
immutable run 的 step 2 恢复并完成至 step 37,330。最终审计结果如下：

- run 目录：`experiments/pi0_full_eef_obs15_act15_v1/20260715T145931Z_pi0_full_eef_f0_obs15_act15_22of25_b33ece09`；
- 数据为全部 22 条执行 episode、`7,465` 个有效 anchor；双卡 DDP，`world_size=2`、global batch=`2`，`steps_per_epoch=3,733`，实际完成 `total_steps=37,330` / 10 effective epochs；
- preflight invocation 为 step `0→2`、`exit_code=0`；固定 747-anchor train-internal monitor loss 为 step 0 `0.57267639`、step 2 `0.57278287`；step 2 train metrics 为 loss=`1.20998967`、lr=`2.997e-08`、grad_norm=`43.70250`；
- 首次 backward 的 missing-gradient ledger 与预先声明的 **exact 6** 个 PaliGemma final-prefix structural parameter tensors 完全一致（合计 `104,861,696` scalar parameters）；overall gradients 全部 finite，vision encoder、action expert、state/action projections 均为 100% tensor/numel coverage，VLM 仅缺上述 exact 6；
- step 2 checkpoint 原子发布，分别保存 `training_state/rank-00/rng_state.safetensors` 与 `rank-01/rng_state.safetensors`；两份 rank-local RNG SHA256 分别为 `d921efb44ccd61c4257e9f3893643e302ea1cb1357447409fc58d56e42c56d7a` 与 `87d35e8305fceaf56927f3022312121eea098cb0f28eafc703cd86fc34d52cee`；
- 正式 invocation 复用同一 run 目录及 checkpoint，日志确认 `start_step=2`、`stop_step=37330`、`total_steps=37330`，进程 `exit_code=0`；最终 checkpoint 为 `step-037330`；
- 固定 747-anchor monitor loss 从 step 0 的 `0.57267639` 降至 step 37,330 的 `0.05408542`，后者也是本轮最优 monitor checkpoint；
- 模型保持 3,238,048,528 个唯一 `requires_grad` 参数、PEFT/LoRA=`0`，完成预训练 π0 的非 LoRA 全参数微调；
- W&B 复用并完成 run `jkrz5th0`；server 端仅有自动生成的 metrics history artifact，`model_artifacts=0`，未上传模型 checkpoint。

固定含义：

```yaml
peft: null
log_freq: 50
policy:
  type: pi0
  pretrained_path: lerobot/pi0_base
  use_peft: false
  freeze_vision_encoder: false
  train_expert_only: false
  dtype: bfloat16
  gradient_checkpointing: true
  chunk_size: 50
  use_relative_actions: false
  normalization_mapping:
    VISUAL: IDENTITY
    STATE: MEAN_STD
    ACTION: MEAN_STD
  push_to_hub: false
wandb:
  enable: true
  project: franka_pi0_full_eef
  entity: null
  mode: online
  disable_artifact: true
  add_tags: true
```

初始训练建议：

| 参数 | F0 |
|---|---|
| train data | `franka_current_eef_obs15_act15_v1` 的全部 22 条 `22of25` PASS episodes |
| observation/action rate | 15 Hz / 15 Hz |
| action chunk | 50 steps，名义 3.33 s |
| optimizer | AdamW（沿用 π0 兼容实现） |
| peak learning rate | `1e-5` |
| final learning rate | `1e-6` |
| warmup | `min(1000, round(0.05 × total_steps))` |
| schedule | cosine decay |
| 最大 epoch | 10 effective epochs |
| seed | 1000 |
| precision | BF16 |
| initial per-device batch | 1，按 S0 实测后只向上调 |
| GPU | S0 单卡；正式训练通过后优先 2×A800 DDP |
| W&B train logging | 每 50 optimizer steps |
| W&B monitor logging | 每个 epoch/checkpoint |

步数在转换后按有效 anchor 计算：

```text
valid_train_anchors = sum(valid_full_horizon_anchor_mask_obs15_act15) = 7465
steps_per_epoch = ceil(valid_train_anchors / global_batch_size)
total_steps = 10 * steps_per_epoch
save_freq = steps_per_epoch
```

DDP 只提高吞吐，不分片单卡模型/optimizer 显存。若 batch size 1 仍 OOM，不允许通过 LoRA 或冻结视觉编码器绕过“全参数”要求；先报告，再评估 FSDP/ZeRO 或更合适资源。

每 epoch 保存本地 checkpoint。由于真实数据没有仿真 environment，`lerobot-train` 的在线 environment `eval_freq` 设为 0；项目本地 evaluator 在固定 `train_monitor_subset` 上计算 `train_monitor_loss`。所有 checkpoint 使用同一 anchor 列表和随机噪声 seed，以保证曲线可比较，并把训练及 monitor metrics 按对应 `global_step` 写入同一个 W&B run。`disable_artifact=true` 只禁止 W&B 上传权重，不影响本地保存和 resume。

### Stage M0：checkpoint 选择与 in-sample 报告

执行状态：**M0_COMPLETE**。最终 checkpoint 与 best monitor checkpoint 均为 step 37,330；
评估覆盖全部 22 条训练 episode、`7,465` 个 full-horizon anchors 和 3 个固定 seeds。
episode-macro mean 结果为：translation ADE/FDE=`12.4826/16.1923 mm`，rotation
ADE/FDE=`1.79266/2.27081 deg`，gripper MAE/FDE=`0.0190087/0.0290417`；所有预测和
指标均为 finite，`nonfinite=0`。完整结果见
`reports/F0_PI0_FULL_TRAINING_22OF25.md` 及对应 M0 JSON 产物。

1. 全部 22 条执行 episode 在整个正式 run 中都属于训练池，不进行 train+val 重训；
2. 结合 training loss、固定 `train_monitor_loss`、数值稳定性和 in-sample Cartesian error 选择 `best_train_monitor`；
3. 对 best/last/milestone checkpoints 在固定监控子集上做一致比较；
4. 对最终 checkpoint 在全部 22 条训练轨迹上生成 teacher-forced in-sample reconstruction 报告；
5. W&B 页面包含连续的 train/monitor loss 曲线，且 artifacts 中没有 model checkpoint；
6. 报告显式标注 `has_held_out_validation=false`、`has_test_set=false`，不输出 validation/test success claim。

### Stage C0：独立的 checkpoint 直接能力评估

执行状态：**CODE READY / NOT RUN**。该阶段按用户补充要求与 F0/M0 审计解耦：只接收
单一 trained checkpoint 和冻结的 action15 训练数据集，显式从模型输入中移除
`action/action_is_pad`，直接执行“双目图像 + current EEF + language → 50-step Cartesian
action chunk”。评估固定覆盖全部 22 条训练 episode、7,465 个 anchors 和 3 个 prediction
seeds；指标严格复用 `reports/F0_PI0_FULL_TRAINING_22OF25.md` §6.2 的口径：ADE/MAE
汇总全部 50 个 waypoints，FDE 使用 waypoint 50，并报告 waypoints `1/5/10/25/50` 的
translation、SO(3) rotation 和 gripper error；先在每条 episode 内汇总，再对 22 条
episode 等权平均，最后对 3 个 seeds 计算 mean 和 population variance。不构造或计算
hold、LOEO、base PI0 等对比基线，也不输出 micro、quantile、tolerance 或 binary accuracy
指标。代码为 `scripts/eval_pi0_direct_capability.py`，详细用法见
`reports/C0_DIRECT_PI0_CAPABILITY_EVALUATION.md`。按照用户要求，本轮只交付代码和用法，
不重新在 F0 checkpoint 上执行 C0。

## 7. 实验矩阵

| ID | 实验 | 目的 | 执行条件 |
|---|---|---|---|
| S0-15 | 15/15 Hz，2–5 step 全参数 smoke | 数据、梯度、显存、保存恢复 | 必做 |
| S1-15 | 15/15 Hz，2 episode 小样本过拟合 | 证明标签和 processor 可学习 | 必做 |
| F0-15 | 15/15 Hz，全部 22 条执行 episode，LR=`1e-5` | 首个归一化、全参数 baseline | 必做 |
| F1-LR | LR=`2.5e-5` | F0 明显欠拟合时比较 π0 默认量级 | 条件执行 |
| R0 | 最优开发配置 seeds=`1000/2000/3000` | 估计小数据训练方差 | 资源允许时推荐 |
| M0 | 固定训练监控子集 + 全训练集重建 | 选择 checkpoint 并形成 in-sample 报告 | 必做 |
| C0 | target-hidden single-checkpoint action-chunk evaluation | 按 F0 §6.2 对全部 22 episodes / 7,465 anchors 计算 all-50 ADE、waypoint-50 FDE、固定 horizons 及 3-seed episode-macro mean/variance | 代码已完成；本轮不执行 |
| S0-30/F0-30 | obs15/action30，`chunk_size=100` | 与真机 30 Hz action cadence 对齐 | 本轮不执行；先复核 F0-15 |

最低执行集合为：

```text
E0 → D0 → D1 → D2 → D3 → D4 → S0 → S1 → F0 → M0
```

不会为了得到更好结果自动扩展实验矩阵。条件实验触发时先在报告中说明证据和额外成本。

## 8. In-sample 监控与 checkpoint 选择

### 8.1 评估协议

对固定 `train_monitor_subset` 中的每个 anchor：

1. 输入真实双目图像和真实 current EEF + gripper；
2. π0 预测完整 50-step、15 Hz action chunk；
3. 用同一 current EEF 解码 predicted waypoints；
4. 与 future measured EEF waypoint proxy 比较；
5. π0 采样固定 prediction seeds；最终候选可对多个固定噪声 seed 取均值；
6. 先按 episode 聚合，再计算全训练集 overall，避免长 episode 支配结果。

正式训练期间用该固定子集比较 checkpoints；最终再对全部 22 条训练 episode 生成同口径结果。这是 teacher-forced、in-sample offline trajectory evaluation，不是 held-out validation，更不是 closed-loop rollout。

### 8.2 指标

必须报告：

- `train_monitor_loss`：固定训练 anchor 和固定随机条件下的 flow-matching loss；
- translation ADE / FDE，单位 mm；
- rotation geodesic ADE / FDE，单位 degree；
- horizon `1/5/10/25/50` 的 translation 和 rotation error；
- gripper MAE；若开合可可靠二值化，再报告 accuracy/F1 和事件时间误差；
- 非有限输出比例；
- 超过 train action 安全分位/硬阈值的比例；
- workspace 越界比例；
- action smoothness；
- 全训练集 overall 和逐 episode 结果；
- 每条 episode 的明细（硬交付，不只报告 aggregate）；
- 固定 prediction seeds 的均值和方差（硬交付）；如执行 R0，再单独报告 training-seed 方差。

主要 checkpoint-selection metric 为固定子集上的 `train_monitor_loss`；train statistics 归一化后的 Cartesian trajectory error 和原始 mm/degree 作为辅助指标。精确计算方式在 D4 后冻结到 monitor config，不根据某个 checkpoint 的结果反向修改监控子集。

### 8.3 结果解释边界

离线结果只能回答：模型是否能拟合已用于训练的 observation 和 realized successful measured trajectory。因为 22 条执行 episode 全部参加训练，它不能证明：

- 真机闭环任务成功率；
- 对观测扰动或新物体布局的恢复能力；
- IK、奇异位形、关节限位和碰撞安全；
- desired EEF 跟踪性能；
- 失败恢复能力，因为首版只含 PASS 演示。

报告首页必须写明：

```yaml
has_held_out_validation: false
has_test_set: false
monitor_subset_source: training_data
generalization_claim: false
```

## 9. 风险与停止/升级规则

| 风险 | 首版处理 | 何时停止或升级 |
|---|---|---|
| future measured EEF/gripper 不等于 desired command | 明确标为 realized waypoint proxy | 若标签滞后导致 S1 不可学习，停止并报告，不伪造 desired |
| current EEF 未完全描述底层构型 | qpos sidecar 仅保留审计信息 | 当前快速拟合实验不加入 qpos；若连 S1/F0 都无法拟合，再提交证据讨论输入定义 |
| cam2/EEF 起止时间不同 | 裁剪到公共有效区间，禁止零图 fallback | 裁剪后任一 episode 不足完整 horizon 或数据损失异常时报告并停止该 episode 转换 |
| 多流不同频率/异步 | cam1 15 Hz 主时间轴、因果 timestamp 对齐、保存 alignment age | source 缺失/过旧或阈值不满足时使 frame/chunk 无效，不按消息序号硬配 |
| 追帧/不规则 dt | v1 保留 cam1 顺序并记录真实 dt | 不作为 v1 阻断项；需要修复时创建 v2，不覆盖 v1 |
| π0 padding loss 未屏蔽 | 15/30 Hz 分别只采完整 50/100-step anchor | 若数据损失不可接受，再实现 masked loss 并单独测试 |
| 30 Hz action 被 stock LeRobot 当成 15 Hz row offsets | 独立 action table、anchor map、dual-rate adapter 与 fail-fast metadata | adapter/偶数 waypoint 对账测试未通过则 30 Hz 数据版本不得发布或训练 |
| `chunk_size=100` 增加显存/计算，且后 50 个 horizon token 超出 π0 base 原 50-step 训练范围 | 本轮只转换/QA 30 Hz 版，先完成 15 Hz baseline | 以后训练 30 Hz 版前单独做 load/forward/backward/OOM 与可学习性 smoke，不自动启动 |
| 没有 held-out validation/test | 明确使用 `train_monitor_loss` 和 in-sample 报告 | 不把下降的监控 loss 解释为泛化或真机成功率 |
| `22of25` 执行数据更少 | 预训练初始化、全量训练、强 QA；保留原始 25 条 source baseline 与 3 条缺失记录 | 结果只用于当前 task 的快速链路验证 |
| 全参数训练 OOM | BF16、checkpointing、先测单卡 | 不改成 LoRA/冻结；先报告 FSDP/ZeRO 方案 |
| 自定义 EEF processor 集成错误 | D2 spike + geometry/unit tests | round-trip、stats 或 reload 不通过则禁止全量转换 |
| 8D carrier stats 被误用于 7D policy action | 独立 `pi0_eef_stats.json` 并显式注入 | stats shape/hash 或 normalized 分布不符即停止 |
| stats 与数据/变换不一致 | 两版分别记录 manifest/dataset/rate/transform/stats hashes | 任一 hash 变化后对应旧 checkpoint 与结果失效；禁止跨 15/30 Hz 混用 stats |
| W&B 鉴权/网络中断 | online mode、run ID 持久化、只由 rank 0 写入 | 不静默关闭或换 run；先停止并报告，确认后再决定 resume/offline sync |
| W&B 意外上传权重 | `disable_artifact=true`，禁止 Artifact/log_model/save | S0 检查发现 model artifact 即停止并修正配置 |
| 需要改 LeRobot 核心 | 项目本地 wrapper 优先 | 确认必须改 `src/lerobot` 时先停并征求同意 |

任何阶段触发停止条件时，保留日志和最小复现，向用户报告“证据、影响、可选方案”，不自行改变输入/action 定义。

## 10. 可复现性与产物记录

每个 run 的 `run_manifest.json` 至少记录：

- 原始 manifest 绝对路径和 SHA256；
- 原始 25 条 source record、3 条缺失项及其 expected path；对执行范围内 22 个 MCAP 记录绝对路径、大小和逐文件完整 SHA256；
- 22of25 scope config、derived manifest、scope-content lock 及其 SHA256；
- 两个转换 config、dataset version、observation/action rate、chunk size 和 dataset metadata hash；
- 两版各自的 `meta/stats.json`、`meta/pi0_eef_stats.json`、30 Hz 版的 `meta/action_30hz_carrier_stats.json`、geometry transform config 的 SHA256；
- 公共有效区间、每流 alignment-age 阈值、source timestamp selector 和 15/30 Hz cross-check hash；
- `train_monitor_subset` anchor 列表、seed 和 SHA256；
- LeRobot Git commit、dirty diff hash、`uv.lock` hash；
- 完整启动命令、resolved config、seed；
- Python/PyTorch/CUDA/cuDNN/Transformers/LeRobot 版本；
- GPU 型号、数量、显存、batch 和峰值显存；
- pretrained checkpoint 标识和本地 revision/hash；
- trainable/total parameter count；
- W&B project、entity、run ID、run URL、mode 和 `disable_artifact=true`；
- best/last checkpoint、选择依据和离线评估版本。

正式运行不覆盖已有 `run_id`。中间 checkpoint 在离线评估完成前不删除。

## 11. 计划中的命令入口

以下是批准后将实现的入口名称，目前不是可执行承诺：

```bash
PROJECT=/m2v_intern/tujiahang/Projects/lerobot/franka_project

uv run python $PROJECT/scripts/audit_raw_data.py \
  --config $PROJECT/configs/data/franka_current_eef_common_v1.yaml

uv run python $PROJECT/scripts/convert_to_lerobot.py \
  --config $PROJECT/configs/data/franka_current_eef_obs15_act15_v1.yaml

uv run python $PROJECT/scripts/convert_to_lerobot.py \
  --config $PROJECT/configs/data/franka_current_eef_obs15_act30_v1.yaml

uv run python $PROJECT/scripts/build_pi0_stats.py \
  --config $PROJECT/configs/data/franka_current_eef_obs15_act15_v1.yaml

uv run python $PROJECT/scripts/build_pi0_stats.py \
  --config $PROJECT/configs/data/franka_current_eef_obs15_act30_v1.yaml

uv run python $PROJECT/scripts/build_train_monitor_subset.py \
  --config $PROJECT/configs/monitor/pi0_eef_v1.yaml

uv run pytest $PROJECT/tests -svv

uv run python $PROJECT/scripts/train_pi0_full.py \
  --config $PROJECT/configs/train/pi0_full_eef_f0.yaml

uv run python $PROJECT/scripts/eval_pi0_train_monitor.py \
  --config $PROJECT/configs/monitor/pi0_eef_v1.yaml
```

环境阶段会把最终可复制命令写入 README 和每个 run manifest；在 D2 custom sampler/processor 通过前，不直接承诺 stock `lerobot-train` CLI 可以覆盖全部 EEF 几何语义。

## 12. 当前执行批次（`22of25` 修订）

本批次已完成：E0–D4、S0、S1、F0 和 M0 全部通过。D0 针对原始 25 条 source manifest 的 `BLOCKED` 结论保留为历史事实；用户随后批准直接使用 22 条完整轨迹继续实验，D1 将范围版本化冻结为 `22of25` 并解锁后续阶段。实际执行顺序为：

1. D3 仅按 derived manifest 转换 22 条，生成 `obs15/action15` 和 `obs15/action30`；
2. 用 `8,793` 主 table rows、`7,465` full-horizon anchors、`373,250/746,500` action stats count 完成硬对账；
3. D4 通过两个数据版本的自动与人工 QA；
4. 仅在 `obs15/action15` 上执行 S0/S1；S0 与 S1 的训练、固定噪声 Cartesian error 和 reload 门禁均已通过；
5. F0-15 双卡 step `0→2` preflight 通过；正式训练从同一 run 的 step 2 恢复并完成至 step 37,330，随后完成 M0 in-sample 报告。

`obs15/action30` 在本轮仍停于 D4，不自动进入训练。D3 与 D4 已按冻结的
`22of25` 范围完成并通过；PI0 本地权重 strict-load/processor preflight 也已通过。
S0 已完成真实 forward/backward、checkpoint、resume、processor reload 和 W&B no-artifact
验收。S1 已从同一 PI0 base 独立完成 300 个 optimizer steps；固定 in-train monitor
loss 从 `0.35633501` 降至 `0.14697757`。在仅含 4 个 train-internal anchors 的固定噪声
评估中，translation/rotation/gripper trained/base ratio 分别为
`0.628869/0.952652/0.500628`，step 300 第二次 strict load 与磁盘 processors reload
得到 normalized/raw action max diff=`0.0`。S1 自身为单次 `0→300` invocation、未执行
resume；resume 能力由 S0 的 `2→3` 链路验证。S1 因此标记 PASS，详见
`reports/S1_PI0_FULL_OVERFIT_22OF25.md`。F0 的双卡 2-step preflight 通过：step 0/2
monitor loss=`0.57267639/0.57278287`，step 2 train loss/lr/grad_norm=
`1.20998967/2.997e-08/43.70250`；首次 backward 仅缺预先登记的 exact 6 个 structural
parameters，两份 rank-local RNG state 均进入 step 2 checkpoint。正式训练复用同一 run
及 W&B `jkrz5th0`，从 `start_step=2` 恢复并完成 `37,330` steps / 10 effective epochs；
固定 monitor loss 最终降至 `0.05408542`。该 run 为 3,238,048,528 个唯一
`requires_grad` 参数的非 LoRA/PEFT 全参数训练。M0 随后达到 `M0_COMPLETE`，覆盖
22 episodes、7,465 anchors 和 3 seeds；episode-macro mean translation ADE/FDE 为
`12.4826/16.1923 mm`，rotation ADE/FDE 为 `1.79266/2.27081 deg`，gripper MAE/FDE 为
`0.0190087/0.0290417`，`nonfinite=0`。W&B run 已完成且 `model_artifacts=0`；完整报告见
`reports/F0_PI0_FULL_TRAINING_22OF25.md`。

## 13. 批准与范围修订记录

### 13.1 首次批准（历史）

```text
计划版本：v1.3-approved
批准状态：APPROVED_FOR_E0_D2
批准人：用户（本会话）
批准时间：2026-07-15T10:28:38Z
需要修改：无；按本计划执行首批 E0-D2
固定 task instruction：stack the cups
W&B：online metrics enabled；model artifact disabled
数据版本：obs15/action15 + obs15/action30；首轮训练仅 obs15/action15
```

### 13.2 `22of25` 执行范围批准记录

```text
计划版本：v1.4-execution-22of25
批准状态：APPROVED_FOR_D3_ON_VERSIONED_22OF25_SCOPE
批准人：用户（本会话）
范围冻结时间：2026-07-15T12:24:52.830049Z
source / included / excluded：25 / 22 / 3
执行边界：D3 数据转换、离线全参数训练与 in-sample 评估
执行假设：same machine；same configured EE frame/TCP
真机 rollout：未授权
固定 task instruction：stack the cups
W&B：online metrics enabled；model artifact disabled
数据版本：obs15/action15 + obs15/action30；首轮训练仅 obs15/action15
```

### 13.3 `22of25` 执行完成记录

```text
计划版本：v1.5-complete-22of25
完成状态：F0_PASS / M0_COMPLETE
完成范围：22 条完整轨迹；obs15/action15 + obs15/action30 均完成，F0 仅训练 action15
F0：PI0 3,238,048,528 requires_grad 参数；PEFT/LoRA=0；37,330 steps / 10 epochs
M0：22 episodes / 7,465 anchors / 3 seeds；nonfinite=0
W&B：run jkrz5th0 finished；model_artifacts=0
结果边界：teacher-forced、in-sample offline evaluation；无 held-out val/test 或真机 rollout
```

## 14. 当前执行状态

| Phase | 状态 | 结论 |
|---|---|---|
| E0 | PASS | Conda、官方 loader、双 A800 BF16/cuDNN/NCCL、pi0_base 和 W&B metrics 通过；W&B artifact=0 |
| D0 | HISTORICAL BLOCKED | manifest 25 条，仅 22 条原始 MCAP 存在；缺失 `00032/00008/00005` |
| D1 | PASS / SCOPE FROZEN | 版本化 `22of25` derived manifest 已锁定；仅该范围获准进入 D3 |
| D2 | PASS | 两条 episode、824 anchors；双频 endpoint、几何、因果时间和 normalization 全部通过 |
| D3 | PASS | 双版本已原子落盘；主 rows=`8,793`、anchors=`7,465`、action stats count=`373,250/746,500`，与冻结目标完全一致 |
| D4 | PASS | 22 episodes 全量时序/sidecar/stats/normalization/video/raw-MCAP QA 通过；最低 PSNR=`38.10 dB` |
| PI0 preflight / graph v3 | PASS | 整理 upstream wrapper heads 后为 3,238,048,528 个唯一 `requires_grad` 参数、PEFT/LoRA=0；其中 6 个 PaliGemma final-prefix 参数（104,861,696）永久在 suffix-only action loss 之外并以 exact ledger 管理，其余 3,133,186,832 参数首次 backward 全部可达；strict 数值加载与官方 state10/action7 processor 通过 |
| S0 | PASS | run `20260715T141209Z_..._13652be2` 完成 3 steps；step 2→3 同 run/W&B resume、checkpoint strict reload、saved processors roundtrip 和真实样本 forward 全部通过；W&B `vay230a2`、artifact=0；详见 `reports/S0_PI0_FULL_SMOKE_22OF25.md` |
| S1 | PASS | 独立 run `20260715T142336Z_..._b8bc6628` 完成 300 steps，固定 4-sample train-internal monitor loss `0.35633501→0.14697757`；translation/rotation/gripper trained/base=`0.628869/0.952652/0.500628`；step 300 model+saved processors reload max diff=`0.0`；W&B `7gnnlevp` finished、artifact=0。S1 未 resume，resume 证据来自 S0；详见 `reports/S1_PI0_FULL_OVERFIT_22OF25.md` |
| F0 | PASS / COMPLETE | 双卡 preflight 后从 step 2 同 run 恢复，完成 `37,330` steps / 10 effective epochs；固定 747-anchor monitor loss `0.57267639→0.05408542`，best/last 均为 step 37,330；3,238,048,528 个唯一 `requires_grad` 参数，PEFT/LoRA=0；W&B `jkrz5th0` finished、`model_artifacts=0`；详见 `reports/F0_PI0_FULL_TRAINING_22OF25.md` |
| M0 | M0_COMPLETE | 全部 22 episodes、7,465 anchors、3 seeds 完成 teacher-forced in-sample 评估；episode-macro mean translation ADE/FDE=`12.4826/16.1923 mm`、rotation ADE/FDE=`1.79266/2.27081 deg`、gripper MAE/FDE=`0.0190087/0.0290417`，`nonfinite=0`；无 held-out val/test 或真机 success claim |
| C0 direct capability | CODE READY / NOT RUN | 独立 evaluator 只评估单一 trained checkpoint；target action 在 processor/policy 前移除，固定覆盖全部 22 episodes / 7,465 anchors，严格按 F0 §6.2 输出 all-50 ADE、waypoint-50 FDE、waypoints `1/5/10/25/50` 和 3-seed episode-macro mean/population variance，不计算对比基线或其他指标；按用户要求未重新运行 F0 |

原 D3 解锁条件为“恢复 3 条缺失 MCAP 并重跑 D0，或由用户明确批准创建带新 version 的 `22of25` 数据范围”。当前通过第二条路径完成解锁；3 条缺失 MCAP 仍保持显式排除，不再阻塞已冻结的 22 条范围。
