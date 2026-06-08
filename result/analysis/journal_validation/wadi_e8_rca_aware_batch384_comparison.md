# WADI e8: RCA-aware Checkpoint 与 Batch Size 对比

日期：2026-06-08  
数据集：`WADI_A1_2017_ds10.csv`  
统一标准：`epochs=8`，`predicted-event`，`prediction_key=15`，RCA 同时报告变量级、子系统级、条件变量级指标。  

## 结论

1. `source-bottleneck-specificity-rca` 仍然是当前 WADI e8 主线。它的变量级 MRR 为 **0.5603**，子系统级 MRR 为 **0.7857**，条件变量级 MRR 为 **0.6048**，整体最均衡。
2. 旧版 `source-bottleneck-rca-aware-checkpoint` 的变量级 MRR 略高，为 **0.5725**，但子系统级 MRR 从 **0.7857** 降到 **0.7440**。这个提升幅度太小，且牺牲了层级解释稳定性，不适合作为主模型。
3. 新增 `source-bottleneck-normalized-rca-checkpoint` 修复了两个严谨性问题：RCA proxy 不再被巨大验证损失数值淹没，并且 proxy 计算后恢复 RNG 状态，避免影响后续训练随机性。但结果没有明显提升：变量级 MRR 为 **0.5679**，低于旧 RCA-aware checkpoint，也低于主模型的条件变量级表现。
4. `--batch-size 192` 只轻微改善检测事件质量，Aff-F1 从 **0.7569** 到 **0.7587**，但变量级 RCA 下降到 **0.5476**，说明小 batch 更像是在改变事件边界，不是改善根因排序。
5. `--batch-size 384` 明确不推荐。变量级 MRR 降到 **0.4718**，Hit@1 降到 **0.3571**，说明“之前效果不够是 batch size 太小导致”的假设不成立。

## 主指标汇总

| 实验 | Profile | Batch | Var-MRR | Var-Hit@1 | Var-Hit@3 | Var-Hit@5 | Subsys-MRR | Subsys-Hit@1 | Cond-Var-MRR | Aff-F1 | Raw-F1@bestAff | 总耗时(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_wadi_e8 | source-bottleneck-specificity-rca | 256 | **0.5603** | **0.4286** | **0.7143** | **0.7143** | **0.7857** | **0.6429** | **0.6048** | 0.7569 | 0.0975 | 1060.9 |
| batch192_wadi_e8 | source-bottleneck-specificity-rca | 192 | 0.5476 | 0.4286 | 0.5714 | 0.7143 | 0.7679 | 0.5714 | 0.5459 | **0.7587** | **0.0988** | 1041.6 |
| rca_aware_ckpt_wadi_e8 | source-bottleneck-rca-aware-checkpoint | 256 | **0.5725** | **0.4286** | **0.7143** | **0.7143** | 0.7440 | 0.5714 | 0.6042 | 0.7570 | 0.0975 | 1146.3 |
| normalized_rca_ckpt_wadi_e8 | source-bottleneck-normalized-rca-checkpoint | 256 | 0.5679 | 0.4286 | **0.7143** | **0.7143** | 0.7500 | 0.5714 | 0.5994 | 0.7570 | 0.0975 | 1131.7 |
| batch384_wadi_e8 | source-bottleneck-specificity-rca | 384 | 0.4718 | 0.3571 | 0.5714 | 0.6429 | 0.7798 | **0.6429** | 0.4613 | 0.7558 | 0.0958 | 1165.5 |

## 判断

RCA-aware checkpoint 的方向在逻辑上合理，因为它尝试让模型选择不只看验证重构误差，也看“是否更利于根因排序”。但 e8 结果说明：当前 proxy 还没有稳定地转化为最终 predicted-event RCA 提升。normalized 版本修复了评价严谨性，却没有带来性能增益，因此该模块目前更适合作为“实验性候选”，不应进入主模型叙事。

主线建议保持 `source-bottleneck-specificity-rca` + batch=256。后续若继续优化，不建议再围绕 batch size 或 checkpoint proxy 小修小补，而应优先研究真正影响根因排序的信息融合机制，例如 source gate、mechanism graph 与 temporal evidence 的深度耦合。

## 产物

- 汇总 CSV：`D:\la_v12\result\analysis\journal_validation\wadi_e8_rca_aware_batch384_comparison.csv`
- 主模型 RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260605_234110_rca.json`
- RCA-aware RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_104619_rca.json`
- Normalized RCA-aware RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_122145_rca.json`
- Batch192 RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_113342_rca.json`
- Batch384 RCA JSON：`D:\la_v12\result\rca\WADI_A1_2017_ds10\20260608_110749_rca.json`
