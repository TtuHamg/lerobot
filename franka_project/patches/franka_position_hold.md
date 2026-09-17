# Franka 等待指令时保持姿态

问题：`safety_gateway` 模式下没有有效模型指令或复位指令时，控制器每周期执行
`q_desired = q_measured`，导致位置误差一直为零，机械臂表现为重力补偿加阻尼。

修复：在进入等待状态时记录一次实测关节位置，并持续使用该位置作为阻抗控制目标。
新的有效模型/复位指令到来后释放保持；再次等待时重新记录位置。控制器重新激活时
清除旧保持目标及复位缓存，从当时的位置开始。保留原来的刚度、阻尼、关节限制和
力矩变化率限制。保持仍是柔顺的关节阻抗控制，不是机械锁止。

适用场景：启动后等待首条指令、disarm 后、模型指令超时后、复位脚本结束后。
不改变模型指令的有效性检查、会话规则或自动 arm 行为。

补丁：[franka_position_hold.patch](./franka_position_hold.patch)，针对外部 ROS 包
`/home/pnp/franka/haply_ros/src/franka_arm_controllers`。可在该包目录以
`patch --dry-run -p1 < /home/pnp/ght_wsp/lerobot/franka_project/patches/franka_position_hold.patch`
检查未修改版本是否适用。安装后的版本再次检查会提示补丁已应用。

验证（2026-09-17）：完整 ROS 2 控制器 Release 编译成功，动态库加载成功；新增的
6 项保持回归测试全部通过。既有 8 项测试中 7 项通过，
`SafetyCommandValidation.EnforcesSessionPlanAndWaypointProgression` 的 2 个断言在
未修改基线与修复版本上均失败，属于已有会话规则与测试预期不一致。本修复未改动它。
尚未做真机验证。

安装记录：源码与动态库已安装，7 个目标文件的 SHA-256 校验通过；ROS 包索引所指的
动态库加载验证通过。原文件备份位于
`/home/pnp/franka/haply_ros/backups/position_hold_20260917_170234/`，其中 `manifest.json`
记录每个目标文件、原文件备份和修改前后的哈希。安装未重启控制器，也未发送机械臂指令。

新动态库必须在控制器进程重启后才会加载。在现场准备好、停止当前任务后，按现有
操作流程重新启动控制栈：

```bash
cd /home/pnp/ght_wsp/lerobot
bash franka_project/scripts/start_joint_stack.sh --execute --no-cam
```

该启动脚本会停止之前的控制栈与客户端，并保持 gateway 未 arm；客户端、相机需按
原来的流程重新启动。新控制器应在等待指令时保持启动姿态。复位仍使用：

```bash
bash /home/pnp/franka/rest_now.sh --min-duration 6 --no-gripper
```

复位完成后应保持停止发布复位指令时的实测姿态。只有需要恢复模型控制时才 arm。
