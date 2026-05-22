# LaGraph v10 (SparseLaGraph) 项目全面理解文档

> **版本**: v10 SparseLaGraph — 极简双图协同异常检测
> **参数量**: ~0.4M (d_model=128, e_layers=2)
> **核心修复**: 点级聚合 (`_point_wise_aggregate`) 解决 raw_f ≈ 0 问题

---

## 项目概览

**LaGraph** 是一个基于**双图协同**的**多变量时间序列异常检测**框架。核心模型 LaGraph v10 (SparseLaGraph) 通过学习**通道自适应图 (C×C)** 和**简化时序图 (L×L)**，结合多尺度重建误差评分实现高精度的异常定位。

项目位于：`/home/professor3/Lagraph/LaGraph`

---

## 项目结构 (v10)

```
LaGraph/
├── ts_benchmark/                          # 核心基准测试框架
│   ├── run_single.py                      # ★ 单次运行入口 (v10 固定配置)
│   ├── pipeline.py                        # 实验流水线调度
│   ├── baselines/self_impl/LaGraph/       # ★ 核心模型 (SparseLaGraph v10)
│   │   ├── LaGraph.py                     # 主类: 单卡训练/推理/损失
│   │   ├── gcn_model.py                   # ★ SparseGCN + MultiScaleAnomalyScorer
│   │   ├── graph_learner.py               # ★ ChannelAdaptiveGraph + SimplifiedTemporalGraph
│   │   ├── temporal_encoder.py            # ★ EncoderStack (2层, d_model=128)
│   │   ├── attention.py                   # OrdAttention (时序自注意力)
│   │   ├── decomp.py                      # MoE 序列分解
│   │   ├── RevIN.py                       # 可逆实例归一化
│   │   ├── channel_mask.py                # 通道掩码
│   │   ├── graph_evolution.py             # (v6 遗留, 已废弃)
│   │   └── vq_bottleneck.py               # (v7 遗留, 已废弃)
│   ├── data/                              # 数据层
│   ├── evaluation/                        # 评估策略与指标
│   ├── report/                            # 报告生成
│   └── ...
├── config/                                # JSON 配置文件
├── dataset/anomaly_detect/                # 数据集
├── result/                                # 实验输出
└── docs/                                  # 文档
```

---

## LaGraph v10 模型架构

### 核心思想

LaGraph v10 是一个**自编码器风格**的异常检测模型，核心创新为**双图协同 + 极简主干**：

1. **ChannelAdaptiveGraph (C×C 通道图)**
   - 先验嵌入 + 数据驱动外积 → 门控融合
   - Top-K 稀疏 (topk=5) + 2层消息传递
   - L1 稀疏正则

2. **SimplifiedTemporalGraph (L×L 时序图)**
   - 可学习位置编码 → Laplacian 归一化卷积
   - 因果偏置温度调制 (causal_encoder → temp)
   - 多尺度时序卷积 + 注意力融合

3. **极简 EncoderStack** (e_layers=2, d_model=128)
   - MultiScaleTemporalConv (dilation=1,3,7,15)
   - OrdAttention + FFN
   - 残差旁路门控融合

4. **多尺度异常评分** (推理阶段)
   - win_sizes=[50, 100, 200] 加权融合
   - 点级聚合: 每个时间点分数 = 覆盖它的所有窗口误差均值
   - 彻底解决 raw_f ≈ 0

### 数据流

```
输入: (B, L, C) 多变量窗口
    │
    ▼
MoE 序列分解 → 残差 (B,L,C) + 趋势 (B,L,C)
    │
    ├──→ ChannelAdaptiveGraph (C×C 自适应通道图)
    │     输出: resid_adapted (B,L,C), A_adaptive (B,C,C)
    │          │
    │          ▼ (A_adaptive 传给时序图)
    ├──→ SimplifiedTemporalGraph (L×L 时序图 + 因果偏置)
    │     输出: resid_temp (B,L,C)
    │          │
    │          ▼
    ├──→ 投影 + 残差旁路门控 → (B,L,d_model=128)
    │          │
    │          ▼
    ├──→ EncoderStack ×2 (时序卷积 + 注意力 + FFN)
    │          │
    │          ▼
    └──→ 投影输出 + 趋势线性 → 重建值 (B,L,C)
              │
              ▼
         损失: MSE 重建 + L1 稀疏正则
         推理: MultiScaleAnomalyScorer → 点级聚合 → 异常分数
```

### 参数量: ~0.4M (v9 的 1/30)

| 模块 | 参数量 | 占比 |
|:---|:---:|:---:|
| ChannelAdaptiveGraph | ~42K | 10% |
| SimplifiedTemporalGraph | ~58K | 14% |
| EncoderStack (2层) | ~235K | 57% |
| 其他 (投影/嵌入/残差) | ~78K | 19% |

### 损失函数 (v10 — 极简)

```
L_total = L_recon + L_sparse

L_recon  = MSE(x_rec, x_input)              # 时域重建（唯一主损失）
L_sparse = λ_l1 · ||A_channel||₁           # L1 稀疏正则
```

**已移除的所有辅助损失项**:

| 损失项 | v10 状态 | 移除原因 |
|:---|:---:|:---|
| L_freq (频域) | ❌ 移除 | 零贡献 |
| L_vq (VQ 量化) | ❌ 移除 | 码书利用率 <0.1% |
| L_contrastive (对比) | ❌ 移除 | InfoNCE 导致嵌入坍塌 |
| L_pred (预测) | ❌ 移除 | 零贡献 |
| L_dag (DAG 约束) | ❌ 移除 | 负优化 |

---

## 固定超参数 (v10 核心设计决策)

**模型容量不随 GPU 数缩放**。RTX 5070 迁移后，当前训练入口固定为单卡路径；`n_gpus>1` 只保留为兼容参数，会自动降级到单卡运行。

| 参数 | v10 值 | 说明 |
|:---|:---:|:---|
| d_model | **128** | 固定 (不受 n_gpus 影响) |
| e_layers | **2** | 固定 |
| n_heads | **4** | 固定 |
| batch_size | **256** | 单卡训练 batch size |
| lr | **1e-4** | 固定 (无 sqrt 缩放) |
| warmup_epochs | **5** | 固定 |
| patience | **15** | EarlyStopping 耐心值 |
| λ_l1 | 1e-5 | L1 稀疏正则权重 |
| win_size | **100** | 滑动窗口长度 |
| topk | **5** | 图稀疏 Top-K |

---

## 训练流程

### 单卡训练 (`_single_gpu_train`)

```
1. 80/20 时间顺序切分 (不打乱)
2. StandardScaler 标准化 (fit on train)
3. DataLoader(step=1, win_size=100)
4. SparseGCN → GPU
5. 优化器: 差分学习率 (图参数 lr×0.1)
6. Warmup(5 epochs) + CosineAnnealing
7. 训练循环: MSE + L1 稀疏
8. EarlyStopping(patience=15) → best checkpoint
```

### RTX 5070 单卡训练

当前主流程已移除 DDP worker 和 `torchrun` 子进程。训练始终在一个 CUDA 设备上运行，避免单卡实验与旧八卡路径出现技术分叉。

**跨数据集 OOM 修复**:
- `_destroy_model_and_clean_cuda()` — 每个数据集训练前彻底清理
- best checkpoint 通过 `EarlyStopping` 保存在 CPU state dict
- 训练结束后再加载最佳权重用于检测和阈值校准

---

## 推理与异常打分 (v10 点级聚合)

### 核心修复: 点级聚合

| 版本 | 方法 | raw_f_score 问题 |
|:---|:---|:---:|
| ❌ v9 | step=win_size 非重叠 + 尾部 padding | 尾部大量 0 → raw_f ≈ 0 |
| ✅ v10 | step=1 完整滑动 + 点级聚合 | 每个点的分数 = 覆盖它的所有窗口的误差均值 |

### 算法 (向量化 O(N))

```
输入: window_scores (N_windows, win_size)  →  step=1 滑动窗口
输出: point_scores (total_length,)

每个点 t 的分数 = Σ(所有覆盖 t 的窗口在对应位置的误差) / 覆盖 t 的窗口数

示例 (win_size=5, total_length=8):
  point 0: 仅窗口0 → 1个分数
  point 2: 窗口0,1,2,3 → 4个分数均值 (最密区域)
  point 7: 仅窗口3 → 1个分数
```

### 多尺度评分 (`multi_scale_forward`)

```
win_size=50  → 重建误差 → max-agg → 权重 1
win_size=100 → 重建误差 → max-agg → 权重 2 (主尺度)
win_size=200 → 重建误差 → max-agg → 权重 1

融合: score = (score_50 + 2×score_100 + score_200) / 4
```

### detect_label (标签预测)

```
1. 训练集: step=1 滑动 → 点级聚合 → train_energy
2. 测试集: step=1 滑动 → 点级聚合 → test_energy
3. combined = concat(train_energy, test_energy)
4. 对每个 anomaly_ratio:
     threshold = percentile(combined, 100 - ratio)
     pred = test_energy > threshold → 二值标签
```

---

## 评估指标对照

| 指标 | 含义 | 与论文关系 |
|:---|:---|:---|
| **Affiliation F1** | 严格点级定位 (边界精度) | ✅ 论文 Table II 的 F1 |
| Adjusted F1 | 区域级 (7点容忍窗口) | ❌ 不是论文 F1 |
| Raw F1 | 精确逐点比较 | v10 期望 > 0.10 |
| AUC/AP | 排序质量 (分数模式) | 辅助指标 |

> ⚠ **重要**: 论文 Table II 的 "F1" = **affiliation_f**。`adjust_f_score` 因 7 点容忍窗口会显著偏高，切勿混淆。

---

## 迁移检查清单 (v9 → v10)

### 必须修复的问题

| # | 问题 | 检查点 |
|:---|:---|:---|
| 1 | d_model 仍按 GPU 缩放 | 确认 `run_single.py` 中 v10 分支覆盖旧逻辑 |
| 2 | e_layers 仍按 GPU 缩放 | 同上 |
| 3 | batch_size 仍为 227 (旧配置) | 确认 `per_gpu_batch=2048` |
| 4 | lr 仍使用 sqrt 缩放 | 确认 `scaled_lr=1e-4` 固定 |
| 5 | 辅助模块仍启用 | SparseGCN `__init__` 已硬编码 False (无需修改) |
| 6 | 点级聚合未启用 | 检查 `_point_wise_aggregate` 是否在 `detect_score/label` 中调用 |
| 7 | 多尺度评分未启用 | 检查 `multi_scale_forward` 是否为默认推理入口 |

### 边界情况

- **SWaT**: win_size=200 时 total_length < 200,000，多尺度评分安全
- **MSL**: 55 维，虽 C≠L，但双图各自独立 → 无冲突
- **SMAP**: 25 维，极小数据集 → 调整 `per_gpu_batch` 不影响模型容量

---

## 常见问题

### Q: 为什么 raw_f ≈ 0 但 adjust_f 很高？

A: **7 点容忍窗口**放大了正确率。如果模型在异常区间内预测了 1 个点，adjust 认为整个区间都正确。v10 的**点级聚合**修复了这个问题。

### Q: 为什么 v10 用 d_model=128 就够了？

A: v9 消融实验证明 d_model=256→512 和 e_layers=3→6 带来 **0 提升**。额外的容量学习到的是训练集的噪声模式，而非泛化的异常特征。

### Q: DDP 还有必要吗？

A: 当前 5070 迁移版不再保留 DDP。模型约 0.4M 参数，单卡已能完成主要数据集训练；移除 DDP 可以避免单卡路径和旧八卡路径在优化器、DataLoader、checkpoint 与评估流程上继续分叉。

### Q: 论文的 F1 怎么复现？

A: 使用 `affiliation_f` 指标。在 `evaluation/strategy/anomaly_detect.py` 中确认 metrics 列表包含 `"Affiliation F1"`。阈值选取使用训练+测试分数联合分位数 (`combined = concat(train_energy, test_energy)`)。

---

*最后更新: 2026-05-22 — RTX 5070 single-GPU SparseLaGraph (0.4M params, 双图协同)*
