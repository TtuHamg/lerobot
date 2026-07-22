# PI0 base 全参数训练图 preflight v2

## 结论

真实 `lerobot/pi0_base` strict-load smoke 已在本地通过；未启动训练、未保存 checkpoint、未创建 W&B run，也未上传 Hub。修复后，优化器看到的每一个唯一参数都属于 PI0 实际扩散训练图。

- PaliGemma `lm_head.weight` 已与 `language_model.embed_tokens.weight` 做 **Parameter 指针级共享**，不再保留 526,647,296 个重复参数。
- action expert 的 `lm_head.weight` 不参与 PI0 forward；现已从项目训练图中删除，不冻结、不交给 optimizer，共移除 263,323,648 个无用参数。
- 修复在 `PI0Policy` 构造后、checkpoint 加载和 optimizer 构造前立即执行。
- 修复后的唯一可训练参数为 **3,238,048,528 / 3,238,048,528**，共 776 个 parameter tensors，无 LoRA/PEFT。
- strict load 最终 `missing=[]`、`unexpected=[]`。

## Strict checkpoint 规则

canonical base checkpoint 只物理保存 PaliGemma tied tensor 的 `lm_head` 名称。loader 现在对 tied alias 做对称处理：只允许 `lm_head` 或 `embed_tokens` 中恰好一个物理 key，shape 必须匹配；两个 key 同时存在会因歧义被拒绝。

base checkpoint 中仍有 canonical expert head：

```text
paligemma_with_expert.gemma_expert.lm_head.weight  [257152, 1024]
```

loader 只会在 key 和 shape 都精确匹配时将其验证后丢弃。shape 不符或任何其他 unexpected key 都会 fail closed。项目 checkpoint 保存时不会写入该 expert head；reload 仍严格通过。项目 checkpoint manifest 额外记录 `training_graph`，resume 会将它与当前 live graph 逐字段比较。

## 真实 smoke 结果

| 项目 | 数值 |
|---|---:|
| GPU | NVIDIA A800-SXM4-80GB（physical GPU 1） |
| strict load + official processors | 137.25 s |
| peak allocated | 19.86 GiB |
| peak reserved | 20.32 GiB |
| unique trainable parameters | 3,238,048,528 |
| parameter tensors | 776 |
| tied duplicate removed | 526,647,296 |
| unused expert head pruned | 263,323,648 |

此次使用最终 action15 数据集的 effective stats：state `10D` count 7,465，action `7D` count 373,250。official PI0 pre/post processor 构建通过。

## 测试覆盖

轻量测试覆盖了 pointer identity、expert head 精确 shape drop/reject、未知 unexpected key 拒绝、Pali tied alias 双向加载、全参数计数，以及项目 checkpoint strict save/reload 和 manifest graph 防篡改。focused 结果：`10 passed`；项目回归：`73 passed`（仅有外部 `mcap_ros2` deprecation warning）。

机器可读结果见 `artifacts/training_preflight/pi0_base_strict_load_smoke_v2.json`。
