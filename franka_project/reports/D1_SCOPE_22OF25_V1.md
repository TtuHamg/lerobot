# D1 Frank3 22of25 范围冻结

## 结论

D1 已通过版本化派生 manifest 将正式 D3 范围冻结为当前存在的 22 条 strict-PASS MCAP。
原始 25 条 manifest 保持只读且未修改；3 条缺失项被显式记录，不使用 viz_cache 替代。
在同一机器并复用采集时 configured EE/TCP 的约束下，D3 数据转换已获准。
本范围锁不授权真机 rollout，也没有执行数据转换或训练。

## 冻结汇总

- Scope：`franka_scope_22of25_v1`
- 状态：`approved_for_d3_conversion`
- source / included / excluded：`25 / 22 / 3`
- duration：`608.013 s`
- cam1 / cam2：`8987 / 9007`
- MCAP 总字节：`17229807405`
- MCAP identity：逐文件 byte size + streamed full-file SHA-256

## 显式排除

| source index | episode | duration s | cam1 | 原因 |
|---:|---|---:|---:|---|
| 15 | `run_20260714_202237_00032` | 29.150 | 435 | `declared_raw_mcap_missing_at_scope_freeze` |
| 22 | `run_20260714_175543_00008` | 45.769 | 685 | `declared_raw_mcap_missing_at_scope_freeze` |
| 24 | `run_20260714_165338_00005` | 15.442 | 201 | `declared_raw_mcap_missing_at_scope_freeze` |

## 纳入的 MCAP 锁

| order | episode | bytes | SHA-256 |
|---:|---|---:|---|
| 0 | `run_20260714_212901_00063` | 1124850283 | `2f18a79f8b457e57e8e4e905d3ac343d308c0a395fa3d7a905ed098e02427f82` |
| 1 | `run_20260714_211745_00057` | 797216205 | `3ee49e6af9fac1fff6b9dd776c899e15d62b8705330909f318d453e5c1586256` |
| 2 | `run_20260714_211529_00055` | 1117758447 | `aec61e003c661f7444920f50c49c6e96a451ed6e8a7c8ad301668c0b33fa5656` |
| 3 | `run_20260714_210915_00052` | 778832020 | `f2be0b5b4c9d64e1a1d8d6dcd0b687ba16c6f4a40e02c1165eb18dad47f1f43c` |
| 4 | `run_20260714_210830_00051` | 656099407 | `b851c9f35e0cfa71c8d71f43b84bcca0bcb4418fda81e1d9631cae5c9d6df4be` |
| 5 | `run_20260714_210537_00050` | 748310898 | `68948f710e1b82fb17e376dbb5e78b3907debfdd085bfce102b10853c015192a` |
| 6 | `run_20260714_210246_00048` | 742447556 | `1c129985e8ec718d060abdbd7de1cfe338d22503b22e0a6e2b72c20d510fa11e` |
| 7 | `run_20260714_205904_00046` | 803290407 | `2ef4ac59b2236cf5557b46db8dbdd831d73a2bec97f3381fac10440fd1e89b57` |
| 8 | `run_20260714_205710_00044` | 849586248 | `2a1903f47313cd739a1326abf5511d2564efffbf36432e80746ebb51c153364a` |
| 9 | `run_20260714_205615_00043` | 782790755 | `a5320b3ca792cd9eab71883d42a1245f94c6ec6a3d4f3c188d34185c41d8c84d` |
| 10 | `run_20260714_204910_00039` | 977913469 | `7e9f8a9e9525164061e25f2181a1be76f584e87a6aa90199151f64e6a564effe` |
| 11 | `run_20260714_204750_00038` | 740009857 | `b3d4840daa89532ce982560e364eb51c3c821d4db0b21746159f65e6c4c97ffa` |
| 12 | `run_20260714_204648_00037` | 805098852 | `94b17c72838211abb6a6f4e6d9745089ff88d4a0ab3e5fd9408494ebcc2a2bd6` |
| 13 | `run_20260714_204515_00035` | 537349035 | `87ebe1438aa92de41936d315b50b3e310efbdd7874821131273f8adad8399702` |
| 14 | `run_20260714_204409_00034` | 714895710 | `e46f048a99aff23ddcdf2b7643fea93a1043256f40420dd452896a58b727e2fd` |
| 15 | `run_20260714_200716_00029` | 941594644 | `d5c56231c4f3df496b7412dae5db578274d10fe3f4f137e844da6ab26a3e3ad5` |
| 16 | `run_20260714_200257_00027` | 1197115502 | `6694acc8013d553681943191b55fb7002afd57de201f3ddabc2abb8fcdc090f8` |
| 17 | `run_20260714_194854_00025` | 646824882 | `3d59ecda50c46ee21d50760968ea32f2871cf9bcf7bc5707ae7aa4cf2e317da4` |
| 18 | `run_20260714_194734_00024` | 777995281 | `ffcc8737b622f874a72b862157570115fccc911ee58acc062b34a14ebbe6e13c` |
| 19 | `run_20260714_193528_00019` | 448346425 | `8e53f7fb52298aa1880a8d752382ac8175ef4bbaf1fdf2c0158b73d1251d16c4` |
| 20 | `run_20260714_181728_00010` | 514728973 | `a718c68db291eca6980bc713a4084b663ab44bbd5182c87e87ccd356b41a6d0b` |
| 21 | `run_20260714_165424_00006` | 526752549 | `8aaa8d5a80138b7ff95920576781860b9bb5183384c7c9a6f68cd8667638fe19` |

## Provenance

- Source manifest：`/ytech_milm/collect_data_103/frank3/20260714/DATA/manifests/train_passed.yaml`
- Source manifest SHA-256：`d2ba4dc819eafd0163ed6673f53ea6e6cc7d492b075978c4db925c5b68dcf38b`
- Scope config SHA-256：`2dd95dcd5b9b953abd816d7863af1b38d92275923af884eadcbd07527273bd56`
- Derived manifest：`/m2v_intern/tujiahang/Projects/lerobot/franka_project/manifests/frank3_train_passed_22of25_v1.yaml`
- Derived manifest SHA-256：`76b839b29840714dc6a0c58261d2709b25dfd846fe7f13761787fd6d02b17a77`
- D0 audit：`/m2v_intern/tujiahang/Projects/lerobot/franka_project/artifacts/raw_audit/d0_raw_audit.json`

## Gate

- D3：仅对本锁中的 22 条允许。
- 原始 25 条 manifest：仍不可被描述为 25/25 可转换。
- 真机：必须保持与采集时相同 configured EE/TCP；本锁本身不授权 rollout。
- 若任一 MCAP size/hash 变化，scope lock 立即失效，需重新冻结并重跑 D0/D1。
