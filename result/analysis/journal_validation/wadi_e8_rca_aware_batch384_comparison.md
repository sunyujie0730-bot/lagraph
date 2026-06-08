# WADI e8: RCA-aware checkpoint 与 batch size 敏感性对比

日期：2026-06-08  
数据集：`WADI_A1_2017_ds10.csv`  
评价协议：predicted-event，`prediction_key=15`，变量级/子系统级/层级 RCA 同一套评价脚本。  

## 结论

1. `source-bottleneck-rca-aware-checkpoint` 对变量级 RCA 有轻微正向作用：`Var-MRR` 从 0.5603 提升到 0.5725，但 `Hit@1/3/5` 没有变化；同时子系统级 `MRR` 从 0.7857 降到 0.7440。因此它不是稳定的大改进，只能作为“checkpoint 选择可影响 RCA”的证据，暂时不建议直接替代主模型。
2. `--batch-size 384` 不支持“原结果差是 batch 太小导致”的假设。变量级 `MRR` 从 0.5603 降到 0.4718，`Hit@1` 从 0.4286 降到 0.3571，条件变量级 `MRR` 从 0.6048 降到 0.4613。
3. batch=384 没有明显提速：本次总耗时约 1165.5s，原 batch=256 约 1060.9s，RCA-aware 约 1146.3s。显存占用更高，但训练效率没有实际收益。
4. 当前建议：继续保留 `source-bottleneck-specificity-rca`、batch=256 作为 WADI e8 主线；RCA-aware checkpoint 可作为后续辅助实验，但需要改进 proxy 权重或 proxy 计算方式后再考虑进入主线。

## 主指标汇总

| 实验 | Profile | Batch | Var-MRR | Var-Hit@1 | Var-Hit@3 | Var-Hit@5 | Subsys-MRR | Subsys-Hit@1 | Cond-Var-MRR | Aff-F1 | Raw-F1@bestAff | 总耗时(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_wadi_e8 | source-bottleneck-specificity-rca | 256 | **0.5603** | **0.4286** | **0.7143** | **0.7143** | **0.7857** | **0.6429** | **0.6048** | 0.7569 | 0.0975 | 1060.9 |
| rca_aware_ckpt_wadi_e8 | source-bottleneck-rca-aware-checkpoint | 256 | 0.5725 | 0.4286 | 0.7143 | 0.7143 | 0.7440 | 0.5714 | 0.6042 | **0.7570** | 0.0975 | 1146.3 |
| batch384_wadi_e8 | source-bottleneck-specificity-rca | 384 | 0.4718 | 0.3571 | 0.5714 | 0.6429 | 0.7798 | 0.6429 | 0.4613 | 0.7558 | 0.0958 | 1165.5 |

## 判断

RCA-aware checkpoint 的变量级 MRR 略高，但提升太小，且子系统级下降，说明当前 proxy 分数还没有稳定对齐真实 predicted-event RCA。它的设计方向合理，但现在的 `source_mrr` proxy 权重 `0.02` 被巨大验证 loss 数值淹没，实际 checkpoint 仍主要由 val loss 决定。

batch=384 的结果更明确：它提高显存占用，却降低变量级 RCA，并没有缩短总耗时。当前 WADI 主线不应扩大 batch；如果后续要测试 batch，只建议小范围测 `192/256/320`，不要优先测更大的 batch。

## 产物

- 汇总 CSV：`D:\la_v12\result\analysis\journal_validation\wadi_e8_rca_aware_batch384_comparison.csv`
- RCA-aware RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_104619_rca.json`
- Batch384 RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_110749_rca.json`
- 新增 CLI 参数：`ts_benchmark/run_single.py --batch-size`
