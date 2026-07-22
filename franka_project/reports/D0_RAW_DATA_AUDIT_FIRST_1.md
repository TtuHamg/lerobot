# D0 Frank3 原始数据审计

- 生成时间（UTC）：`2026-07-15T10:43:38.899796+00:00`
- Manifest：`/ytech_milm/collect_data_103/frank3/20260714/DATA/manifests/train_passed.yaml`
- Manifest SHA256：`d2ba4dc819eafd0163ed6673f53ea6e6cc7d492b075978c4db925c5b68dcf38b`
- Episode：`1` 条严格 PASS
- 原始 cam1 帧：`588`
- 公共区间 cam1 anchors：`587`
- 15 Hz 有效完整 horizon anchors：`468`
- 30 Hz 有效完整 horizon anchors：`468`

## 关键结论

- 公共有效区间按 cam1/cam2/EEF/gripper 起止覆盖裁剪，未使用零图 fallback。
- 15 Hz 与 30 Hz carrier 的 endpoint value/source timestamp 全量交叉对账通过。
- EEF frame_id 集合为 ['base']；joint/gripper name order 在 JSON 中逐条记录。
- 配置阈值下共有 4 个 observation 因 source 过旧而无效。
- 15/30 Hz 有效 full-horizon anchor 集合数量一致。

## Aggregate alignment age（ms）

| stream | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| cam2 | 35.093 | 45.328 | 45.917 | 168.264 |
| eef | 14.491 | 31.956 | 33.014 | 33.221 |
| gripper | 1.035 | 1.909 | 2.014 | 2.117 |
| qpos_audit | 0.555 | 1.013 | 1.114 | 1.665 |

## Episode 明细

| episode | raw c1/c2 | common M | invalid obs | anchors 15/30 | cam2 max ms | EEF max ms | cam1 max dt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| run_20260714_212901_00063 | 588/587 | 587 | 4 | 468/468 | 168.264 | 33.221 | 78.322 |

## 限制

- 本报告只审计数据完整性、时间语义和离线标签可构造性，不代表真机闭环成功率。
- 使用 MCAP `log_time_ns` 作为本轮同步时钟；header/log latency 同时保存在 JSON。
- `future measured EEF/gripper` 是 realized waypoint proxy，不是采集控制器的 desired command。

