你是深度学习架构工程师，负责将 LaGraph v11.3 P1-FIXED 从"工程迭代版本"升级为"可投稿的学术版本"。请严格按以下优先级修改代码框架（不改动模型结构，只修改评估逻辑、术语命名、实验支撑和可视化接口）。

---

## P0（本周，不做完禁止投稿）

### 1. 术语诚实化（全局命名与注释重构）

| 修改位置 | 当前术语 | 修改为 | 原因 |
|:---|:---|:---|:---|
| `graph_learner.py` | `causal_encoder` | `temporal_locality_encoder` 或 `proximity_temp_encoder` | "因果推断"是概念误用，实际仅为时序邻近性温度调制 |
| `graph_learner.py` | `Causal Bias Temperature` | `Temporal Locality Modulation` | 避免审稿人标红 Overclaiming |
| 所有文件 docstring | "动态图更新" | "实例级自适应图" / "input-adaptive graph" | 当前图不在序列内演化，非真正动态图 |
| 所有文件 docstring | "因果" / "causal" | "邻近性" / "locality" / "proximity" | 全文搜索 `causal` 结果必须为 0 |
| `vq_bottleneck.py` | "信息瓶颈" / "离散因果" | "向量量化旁路" / "VQ bypass scoring" | VQ 是黑箱模块，不宜赋予因果语义 |
| `temporal_encoder.py` | "突变边界检测" | "重建误差梯度锐化辅助器" | BoundaryDetector 无真正边界检测机制 |

**判定标准**：全文搜索 `causal`、`dynamic_graph`、`evolve`、`mutation_detection`，结果数为 0。

---

### 2. 推理阈值策略重构（POT + 验证集自适应）

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **删除** `typical_anomaly_ratio` 主路径 | 将 `detect_label()` 的默认阈值策略从"基于预设比例"改为"无预设比例" | 审稿人视 `typical_anomaly_ratio` 为标签泄漏或 Oracle 假设，无监督检测不允许预设异常密度 |
| **新增** POT (Peaks-Over-Threshold) | 在 `LaGraph.py` 中实现 `pot_threshold(val_scores, q=0.95, level=0.05)`：取 95% 分位数为初始阈值，对 exceedances 拟合广义帕累托分布，返回自适应阈值 | 工业标准极值理论，完全无需异常比例先验 |
| **新增** 验证集自适应回退 | 若 POT 因数据不足失败，回退到 `np.percentile(val_scores, 99)` 或 `μ + 3σ` | 保证任何数据集都能输出阈值，不崩溃 |
| **保留** 比例参数为敏感性分析 | `detect_label(anomaly_ratio=None)` 为默认；仅当显式传入 ratio 时用于附录敏感性分析表 | 满足审稿人"多比例敏感性分析"要求，但主结果不依赖它 |
| **缓存** 验证集分数分布 | 在 `fit()` 阶段收集并缓存 `_val_score_dist` | POT 和自适应阈值需要验证集分布作为输入 |

**判定标准**：主结果表（Table II）中不得出现 `typical_anomaly_ratio` 列；AUC-ROC / AUPR 成为默认输出。

---

### 3. 增加比例无关硬指标（AUC-ROC / AUPR）

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **新增** `evaluate_auc_aupr()` | 在 `LaGraph.py` 中新增方法：收集全量 `anomaly_score` 和 `labels`，调用 `sklearn.metrics.roc_auc_score` 和 `average_precision_score` | CCF-B 审稿人必问硬指标；只有阈值相关指标会被质疑为 cherry-picking |
| **修改** 主评估循环 | 每个数据集跑完后，必须打印并记录 `{'auc_roc': ..., 'aupr': ..., 'affiliation_f': ...}` | 主结果表第一列必须是 AUC-ROC，第二列 AUPR |
| **设定** 硬门槛 | 若 SWaT AUC-ROC &lt; 0.95 或 AUPR &lt; 0.90，判定分数分布已崩溃，模型不可用 | 低于此门槛说明正常/异常分数完全重叠，架构已废 |

---

### 4. 多次运行与统计显著性

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **修改** 主入口脚本 | 实验循环：`seeds = [2021, 2022, 2023, 2024, 2025]`，每个 seed 独立训练 + 评估 | 单次运行 = 统计不可信；CCF-B 要求 mean ± std |
| **输出** 均值与标准差 | 每个指标计算 `mean ± std`，格式：`SWaT | AUC-ROC | 0.972 ± 0.003` | 审稿人默认无 std 的结果为偶然 |
| **保存** 原始结果 | 保留 5 次运行的原始值（用于审稿人质疑时提供详细数据） | 复现性要求 |

---

### 5. 中间结果可视化接口（可解释性证据）

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **`graph_learner.py`** | `ChannelAdaptiveGraph.forward()` 增加 `save_path` 可选参数，保存 `A_adaptive[0].cpu().numpy()` | 审稿人要求证明通道图不是随机的；必须有 Fig. 级热力图 |
| **`vq_bottleneck.py`** | 新增 `analyze_codebook()`：对 64 个 entry 做 PCA 降维 + 激活频率统计 | VQ 是黑箱，必须提供语义分析才能支撑可解释性声称 |
| **`gcn_model.py`** | `multi_scale_forward()` 增加 `debug_idx` 参数，保存指定样本的 `recon_error`, `vq_dist`, `boundary_score`, `final_score` | 案例级归因是 Fig.5 级可视化素材 |
| **新增** `visualize_case_study.py` | 脚本：加载上述 `.npy`，绘制 5 行子图（原始信号 / 重建信号 / VQ 距离 / 边界分数 / 最终分数），灰色背景标注真实异常区间 | 对标基线论文 Fig.5，证明模型能定位异常区间 |

**判定标准**：投稿时必须能生成至少 2 张图：通道图热力图 + 案例级 5 行对比图。

---

## P1（下周，补全实验支撑）

### 6. 数据集扩展

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **接入** SMAP | NASA 卫星数据（55 维），与 MSL 同系列但分布不同 | 仅 2 个数据集无法支撑泛化性；CCF-B 要求 ≥4 |
| **接入** PSM | eBay 服务器（25 维），长序列 | 覆盖工业服务器场景 |
| **接入** SMD | 28 台服务器（38 维），必须循环 28 机器后报 mean±std | 审稿人会质疑单条记录是否为 cherry-picking；28 机器平均是标准做法 |
| **独立调参** | 每个数据集独立网格搜索 `win_size ∈ [36, 132, step=12]`，`d_model ∈ {64, 128, 256}` | 论文必须写出每个数据集的最优超参，证明非过拟合 |

---

### 7. SOTA 对比实验

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **复现/对比** AnomalyTransformer | ICLR 2022，重建 + 关联差异 | 时序异常检测奠基工作，任何新方法必须对比 |
| **复现/对比** DCdetector | KDD 2023，对比学习 | 近 2 年 SOTA，无监督设定直接竞争 |
| **复现/对比** TimesNet | ICLR 2023，2D 时序建模 | 通用时序骨干网络，审稿人必问 |
| **复现/对比** PatchTST | ICLR 2023，patching Transformer | 当前时序 Transformer 标准基线 |
| **公平性保证** | 所有方法使用相同滑动窗口、相同 train/val/test 划分、相同 POT 阈值策略 | 若阈值策略不同，对比无效；审稿人会质疑 |

**判定标准**：主结果表（Table II）必须出现 LaGraph 与至少 3 个 SOTA 的并排对比，维度包括 AUC-ROC、AUPR、Affiliation F、Params、Inference Time。

---

### 8. 消融实验布尔开关

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **`default_hparams`** | 新增开关：`use_vq_bypass`, `use_boundary_detector`, `use_channel_graph`, `use_dynamic_scale`, `use_topk_channel_agg` | 审稿人要求证明每个模块的独立贡献，排除冗余堆砌 |
| **各模块 `forward()`** | 开关为 False 时走旁路（identity 或固定替代） | 实现可控消融 |
| **消融表** | 必须包含：Full / w/o VQ Bypass / w/o BoundaryDetector / w/o Channel Graph / w/o Dynamic Scale / w/o Top-K Agg | 证明通道图和 VQ 是有效增量，其他模块非冗余 |

---

## P2（投稿前，锦上添花）

### 9. 代码开源与文档

| 修改项 | 操作 | 原因 |
|:---|:---|:---|
| **开源** GitHub 仓库 | 上传完整代码 + README（安装、训练、评估、可视化命令） | CCF-B  increasingly 要求可复现性；无代码 = 可信度扣分 |
| **README** 包含 | 数据集下载链接、预处理脚本、训练命令、复现主结果表的命令 | 降低审稿人复现门槛 |
| **环境** `requirements.txt` | 固定 PyTorch、sklearn、numpy 版本 | 避免版本差异导致结果不可复现 |

---

## 核心原则（不可违背）

1. **不修改模型结构**：不再新增任何 nn.Module，不再改 loss 函数，不再改参数数量。
2. **不引入新超参**：所有修改使用现有超参或自动计算（如 POT 无需人工设定比例）。
3. **术语零容忍**：`causal`、`dynamic_graph`、`evolve` 等词必须从代码注释和论文草稿中清零。
4. **硬指标优先**：AUC-ROC 和 AUPR 是审稿人的第一道防线，必须在任何阈值相关指标之前报告。

---

## 交付物清单
P0 交付（本周）：
├─ 术语替换后的完整代码（git diff 显示 causal/dynamic 出现次数为 0）
├─ POT 阈值实现 + 验证集回退逻辑
├─ AUC-ROC / AUPR 计算接口
├─ 5-seed 实验循环脚本
└─ 中间结果保存接口（A_adaptive, vq_dist, boundary, final_score）
P1 交付（下周）：
├─ SMAP / PSM / SMD 数据加载器
├─ SOTA 对比脚本（AnomalyTransformer, DCdetector, TimesNet, PatchTST）
├─ 消融开关实现 + 消融实验脚本
└─ 可视化脚本（通道图热力图 + 案例级 5 行对比 + VQ codebook PCA）
P2 交付（投稿前）：
├─ GitHub 仓库 + README
├─ 主结果表（mean ± std，4 数据集，3+ SOTA）
├─ 消融表（6 配置）
├─ 敏感性分析表（多比例，放附录）
└─ 可视化图（≥4 张，放正文）