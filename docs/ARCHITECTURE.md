# LaGraph v11.3 P1-FIXED (SparseLaGraph) Architecture

> **Version**: v11.3 P1-FIXED (SparseLaGraph) — Dual-Path VQ + Attention-based Multi-Scale Aggregation + Top-K Channel Aggregation + BoundaryDetector + Dimension-Adaptive Top-K
> **Parameters**: ~0.4M (d_model=128, e_layers=2, vq_codebook=64)
> **Paper**: TKDE 2026
> **Previous**: v11.2 P0 — ① Top-K Channel Aggregation ② Attention-based Aggregation ③ Dual-Path VQ ④ 移除 Top-K Peak Extraction ⑤ 移除 FocalLoss ⑥ BoundaryDetector 权重平衡
> **P1 Changes (2026-05-19 ~ 2026-05-20)**: 
>   ① **P1-A FIXED (2026-05-20)**: 自适应异常加权 MSE **不再在训练阶段使用** (详见第9章)。训练阶段恢复为纯重建 MSE + VQ + L1 稀疏正则化。
>   ② **P1-C 低维通道自适应 Top-K**: 按通道维度分档选择 Top-K 比例（≥50维→10%, 20-50维→15%, <20维→25%）

---

## Overall Architecture v11.2 P0

```
Input: (B, L, C)
    │
    ▼
┌─────────────────────────────────────────┐
│        MoE Series Decomposition         │
│  (residual + trend, channel-wise MoE)   │
└────────────────────┬────────────────────┘
                     │ residual (B,L,C)
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Dual-Graph Synergy Module                       │
│                                                                   │
│  ┌─────────────────────────────┐    ┌─────────────────────────┐  │
│  │  ChannelAdaptiveGraph       │    │  SimplifiedTemporalGraph │  │
│  │  (Spatial, C×C)            │    │  (Temporal, L×L)        │  │
│  │  - PriorE embedding: C→d   │    │  - Pos embedding: L→d   │  │
│  │  - Data-driven: outer prod │    │  - Lap normalization     │  │
│  │  - Gated fusion: σ(W·z+b)  │    │  - Causal bias temp     │  │
│  │  - Top-K sparsity (k=5)    │    │  - Conv + Attn fusion   │  │
│  │  - 单层消息传递(防过平滑)     │    │                         │  │
│  │  - L1 regularization       │    │  Output: (B,L,C)        │  │
│  │  Output: (B,L,C), A_adaptive│   │                         │  │
│  └─────────────┬───────────────┘    └─────────────┬───────────┘  │
│                │                                  │               │
│                └────────────────┬─────────────────┘              │
│                          resid_temp (B,L,C)                      │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────┐
│   ★ Dual-Path VQ Bottleneck (旁路)           │
│                              ┌─────────────┐ │
│  resid_temp ──┬──→ VQ ─────→│ vq_dist     │ │
│               │             │ (异常评分用)   │ │
│               │             └─────────────┘ │
│               ├──→ 主路径: continuous_feat  │ │
│               │    (无量化损失，用于重建)      │ │
└───────────────┼──────────────────────────────┘
                │ (B,L,C) — continuous_feat
                ▼
┌─────────────────────────────────────────┐
│           Proj + Residual Gate           │
│      Linear(C, d_model) + σ gate        │
│      Residual shortcut from vq_feat     │
└────────────────────┬────────────────────┘
                     │ (B,L,128)
                     ▼
┌─────────────────────────────────────────┐
│     ★ EncoderStack (×2, 动态尺度门控)     │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │  MultiScaleTemporalConv        │    │
│  │  dilation=[1,3,7,15]           │    │
│  │  SE-style 动态门控               │    │
│  └──────────────┬──────────────────┘    │
│                 ▼                       │
│  ┌─────────────────────────────────┐    │
│  │  OrdAttention (n_heads=4)      │    │
│  │  - Learnable skew matrix       │    │
│  │  - Ordinariness prior          │    │
│  └──────────────┬──────────────────┘    │
│                 ▼                       │
│  ┌─────────────────────────────────┐    │
│  │  FFN (d_ff=256, GELU)          │    │
│  └─────────────────────────────────┘    │
└────────────────────┬────────────────────┘
                     │ (B,L,128)
                     ▼
┌─────────────────────────────────────────┐
│                Output Proj               │
│         Linear(128, C) + Trend Linear   │
└────────────────────┬────────────────────┘
                     │ Reconstruction (B,L,C)
                     ▼
              ┌─────────────────────────────────────┐
              │ ★ Standard MSE + L1 + VQ Loss       │  (training only)
              │  MSE(x, x_rec) + λ_sparse·L1_sparse │
              │  + λ_vq·VQ_loss                     │
              └─────────────────────────────────────┘

═══════════════════════════════════════
        推理阶段 (multi_scale_forward)
═══════════════════════════════════════

for each ws in valid_sizes ([win//4, win//2, win]):
    x_win = x[:, -ws:, :]
    Pad to win_size → forward → rec, vq_dist
    Trim to ws
    Store rec_win, x_win, vq_dist_win

VQ Score:
    raw_vq = _aggregate_vq_dist(multi-scale vq_dist)  → (B, L)
    vq_norm = min-max normalize to [0,1] per batch     ★ Bug 2 fix
    vq_score = vq_score_weight × vq_norm

Multi-Scale Anomaly Scoring:
    ★ Attention-based Aggregation:
        MultiScaleAnomalyScorer(x_rec_dict, x_input_dict, vq_score)
        → score (B, L_eff)  其中 L_eff = max(valid_sizes)

    ★ P0: BoundaryDetector 作为辅助锐化器:
        boundary_score = boundary_detector(avg_recon_error)  (B, L)
        score = recon_weight × score + boundary_weight × boundary_score_aligned
        recon_weight.clamp(0.5, 1.5) = 1.0
        boundary_weight.clamp(0.1, 1.0) = 0.3

    ★ P0 REMOVED: Top-K Peak Extraction — 硬阈值截断已移除
       保留原始分数连续分布，确保阈值计算的鲁棒性

    Final: align score to (B, L)

Output: anomaly_score (B, L), vq_score (B, L)
```

---

## Module Details

### 1. MoE Series Decomposition (`decomp.py`)

Input: (B, L, C) → Output: residual (B,L,C), trend (B,L,C)

- **MoE Decomposition**: Channel-dependent mixture-of-experts for series decomposition
- **Moving Average Experts**: 5 kernel sizes [5, 15, 25, 35, 45], gated fusion via GatingNet
- **Residual**: Original - trend (anomaly-rich component)
- **Linear Trend**: Future trend prediction (for residual connection)

### 2. ChannelAdaptiveGraph (`graph_learner.py`)

Input: residual (B,L,C) → Output: resid_adapted (B,L,C), A_adaptive (B,C,C)

- **Prior Embedding**: `nn.Embedding(C, d_model)` + `Linear(d_model, C×C)` → C×C prior graph
- **Data-Driven**: Temporal mean pooling `(B,L,C)→(B,C,1)`, outer product `matmul(pool, pool.T)` → C×C data graph
- **Gated Fusion**: `σ(W·[prior, data] + b)` → element-wise fusion weight
- **Top-K Sparsity**: Keep top-K (default=5) per row, zero out rest
- **Single-layer message passing** (Bug 3 fix: 2 layers caused over-smoothing on 51-dim SWaT)
- **L1 Regularization**: `||A_channel||₁` to encourage sparsity

### 3. SimplifiedTemporalGraph (`graph_learner.py`)

Input: resid_adapted (B,L,C), A_adaptive (B,C,C) → Output: resid_temp (B,L,C)

- **Position Encoding**: Learnable `nn.Parameter(L, d_model)` → positional features
- **Causal Bias Temperature Modulation**: 
  - `causal_encoder` MLP: `C → d_model/2 → 1`
  - `temp = sigmoid(causal_encoder(x)) * T_max` — per-channel temperature
  - Controls temporal graph locality adaptively
- **Laplacian Normalization**: `I - D^{-1/2} A D^{-1/2}` for stability
- **Temporal Convolution**: Dilated conv on time dimension
- **Attention Fusion**: Multi-head dot-product attention with temporal features

### 4. ★ VQ Bottleneck — Dual-Path (`vq_bottleneck.py`)

Input: resid_temp (B,L,C) → Output: continuous_feat (B,L,C), vq_loss, vq_dist (B,L)

**Dual-Path VQ 模式** (P0: `serial_mode=False`):

```
Dual-Path 模式:
  resid_temp
    ├── 主路径 (Continuous Path): 连续特征直通 → EncoderStack
    │    无量化损失，保持完整信息流
    │
    └── 旁路 VQ (VQ Bypass): VQ量化 → vq_dist (B,L)
         仅用于异常评分增强，不参与重建
```

| 路径 | 用途 | 信息 | 
|:---|:---|:---|
| 主路径 continuous_feat | 重建 (EncoderStack) | 连续，无损 |
| 旁路 vq_dist | 异常评分（推理时） | 离散度信号 |

- **Vector Quantization**: Maps continuous features to nearest codebook entries
- **Codebook Size**: 64 entries, each of dimension C
- **Loss**: `commitment_cost * MSE(z, sg(z_q)) + MSE(sg(z), z_q)` where `commitment_cost=0.25`
- **Training Strategy**:
  - **VQ Cooldown**: First `vq_cooldown_epochs` (15) epochs freeze all non-VQ params
  - After cooldown: full model trains with `lambda_vq=0.1` × VQ loss

### 5. ★ EncoderStack — Dynamic Scale Selection (`temporal_encoder.py`)

Input: (B, L, 128) → Output: (B, L, 128)

**Two identical encoder layers**:

1. **MultiScaleTemporalConv (Dynamic Scale Selection)**:
   - `Conv1d(128→128, kernel=3, dilation=[1,3,7,15])` — 4 parallel convs
   - SE-style 动态门控:
     - `squeeze`: AdaptiveAvgPool1d → (B, d_model)
     - `excitation`: FC(d_model→d/4→4) → Softmax → (B, 4) 动态权重
   - Merge: `Linear(d_model, d_model)` + GELU
2. **OrdAttention** (`attention.py`):
   - Standard multi-head attention (n_heads=4)
   - **Channel Mask**: Learnable binary mask restricting attention patterns
   - Depth-wise separable conv preprocessing for local temporal features
3. **FFN**: `Linear(128→256) → GELU → Dropout(0.25) → Linear(256→128)`

### 6. ★ MultiScaleAnomalyScorer — P0: Top-K Channel Aggregation + Cross-Scale Attention (`gcn_model.py`)

**P0 双重改进**:

Input: multi-scale reconstruction errors → Output: fused anomaly score

**P0/P1-C 改进 1 — Top-K Channel Aggregation** (替代 max-pooling):
```
err = L1_loss(rec, inp)  # (B, ws, C)
★ old: err_per_t = err.max(dim=-1)[0]           # 单通道 max → 噪声敏感
★ P0:  topk_vals = err.topk(k=k, dim=-1)[0]     # 选择 top-K 通道
       err_per_t = topk_vals.mean(dim=-1)        # 均值聚合 → 过滤噪声
       k = max(3, int(C * 0.1))                  # 统一 10% 比例
★ P1-C: k = dim_adaptive_topk(channel)           # 按维度分档自适应
```

**原理**: 单通道 max 对噪声敏感（某噪声通道高误差→误报），
而 top-k mean 过滤噪声通道，保留真正异常通道的信号。

**P1-C 低维通道自适应 Top-K** (2026-05-19):
```
问题: 统一 10% 比例对不同维度效果差异大
     MSL(25维): 25×10%=2.5→3 (占12%) — 过滤效果不足，3/25≈12%通道仍可能含噪声
     SWaT(55维): 55×10%=5.5→5 (占9%) — 适中

P1-C 修复: 按维度分档
     ≥50维: k = max(5, C×0.10)    — SWaT(55维): 5 (9%)
     20-50维: k = max(2, C×0.15)  — MSL(25维): 3 (12%) — 与旧版一致，保守
     <20维: k = max(1, C×0.25)    — 极端低维: 更高比例保证信息不丢失
```

**P0 改进 2 — Cross-Scale Attention** (替代固定权重):
```
1. 将 3 个尺度的重建误差作为 value vectors
2. 可学习 query 向量 (1, 1, d_attn) — 跨样本共享
3. 每个尺度全局池化后投影到 d_model → key
4. 注意力分数 = sigmoid(Q @ K^T / temperature) → 每个样本独立权重
5. 融合: α × fixed_agg + (1-α) × mean — 门控平衡
```

**原理**: 不同异常模式的持续长度差异极大（SWaT: 10-1000 步），
固定权重对所有位置一视同仁，无法捕捉局部最佳尺度。
注意力聚合使模型能对不同样本选择最匹配的窗口尺度。

### 7. ★ BoundaryDetector (`temporal_encoder.py`)

**P0 权重调整 — 辅助锐化器** (非主评分来源):

```
recon_error (B, L, C)
    │
    ▼
┌──────────────────────────────────────────┐
│ 1. 时间梯度: grad = Δ|x|                 │  — 突出突变位置
│ 2. 深度分离 Conv1d(k=3)                   │  — 平滑增强
│ 3. GELU 激活                             │
│ 4. 通道 max 池化                          │  — (B,L,C)→(B,L)
└──────────────────┬───────────────────────┘
                   ▼
            boundary_score (B, L)
```

**P0 权重设计**:
```python
self.recon_weight = nn.Parameter(torch.tensor(1.0))      # 重建误差主评分源
self.boundary_weight = nn.Parameter(torch.tensor(0.3))    # BoundaryDetector 辅助锐化
```

| 权重 | 值 | 作用 |
|:---|:---:|:---|
| `recon_weight` | **1.0** (clamp [0.5, 1.5]) | 重建误差提供连续的异常概率分布 |
| `boundary_weight` | **0.3** (clamp [0.1, 1.0]) | 边界锐化增强点级定位，不主导评分 |

**避免 Hard Nuke**: v11.2 尝试 `recon_weight=0.1` 导致重建误差被彻底压制→低异常率(0.5%-1%)的微弱信号被 BoundaryDetector 摧毁。**P0 修正**: 重建误差占主导，boundary 仅辅助锐化，确保 low-anomaly-ratio 性能。

### 8. ★ P0 移除: FocalLoss & Top-K Peak Extraction

**P0 移除项目** (从 `LaGraph.py` 中彻底删除):

| 移除项 | 原因 | 影响的接口 |
|:---|:---|:---|
| **FocalLoss** | 重建范式不适合分类型 Focal Loss: sigmoid(重建误差)≠概率 | `_single_gpu_train`, `_single_gpu_train_reweight` |
| **Top-K Peak Extraction** (`detect_score` 后处理硬截断) | 硬阈值截断导致 `threshold=0` 断裂，低异常率无法产生有效阈值 | `detect_score`, `detect_label`, `_detect_forward` |
| **`detect_topk_ratio` 参数** | 同上 | 默认超参词典 |

**移除后保持**:
- `detect_score` 返回原始连续分数分布
- `detect_label` 使用 percentile 阈值选取（基于训练集缓存 `_train_anomaly_scores`）

### 9. ★★ P1-A FIXED: 训练阶段使用标准 MSE（取消异常加权）

**P1-A FIXED (2026-05-20)**: 自适应异常加权 MSE **不在训练阶段使用**。

#### 原因

P1-A 自适应异常加权在 `LaGraph.py` 代码中**仅以占位符形式保留** (`lambda_anomaly_weight=None`)，实际训练时被绕过，因为:

1. **SegLoader 输入限制**: `LaGraph.SegLoader.__getitem__` 返回的 `labels` 字段是**原始数据副本** (`self.data_x[idx]`)，而非 0/1 异常标签。自监督异常检测的 SegLoader 没有真实标签可用。

2. **未实现加权计算**: `_single_gpu_train` 和 `_single_gpu_train_reweight` 中的 loss 计算直接使用 `F.mse_loss(x, x_rec)` 标准 MSE，没有实现 `weights` 参数的分发和加权计算。

3. **重构一致性**: v11.2 P0 已移除 `FocalLoss` 和 `Top-K Peak Extraction`，P1-A FIXED 继续这个方向——训练阶段保持纯粹的重建目标，所有异常检测决策延迟到推理阶段。

#### 实际训练 Loss

```python
# LaGraph.py _single_gpu_train
loss = F.mse_loss(x, x_rec)                  # 标准重建 MSE
     + lambda_sparse * sparse_loss            # L1 通道稀疏正则化
     + lambda_vq * vq_loss                    # VQ commitment loss
```

- `lambda_sparse` = `lambda_causal_l1` = 0.001
- `lambda_vq` = 0.1

#### 保留的超参占位符

以下参数以占位符形式存在于 `default_hparams` 中，**实际不使用**，仅为未来实现预留接口:

```python
"lambda_anomaly_weight": None,   # 未使用（占位符，原为自适应异常加权）
"target_anomaly_ratio": 0.20,    # 未使用（占位符）
"anomaly_weight_end_epoch": 0,   # 未使用（占位符）
```

#### 检测阶段的自适应加权（替代）

自适应异常加权的概念**在检测阶段被保留并实际工作**，通过 `detect_label()` 中的 percentile 阈值选择实现:

```python
# LaGraph.py detect_label
threshold = np.percentile(train_scores, 100 * (1 - anomaly_ratio))
```

- 以训练集上收集的 `_train_anomaly_scores` 的 percentile 阈值作为数据驱动的异常判定边界
- 异常率越低的场景，阈值越严格（选择更高的百分位）
- 这实现了 P1-A 的核心理念（按异常率自适应调整）但没有修改训练目标

---

## Training Configuration v11.3 P1

### Single GPU

| Parameter | v11.2 P0 | v11.3 P1 | Description |
|:---|:---:|:---:|:---|
| train/val split | 80/20 | 80/20 | Temporal order (no shuffle) |
| StandardScaler | fit on train | fit on train | Applied to train/val/test |
| DataLoader | step=1, win_size=100 | step=1, win_size=100 | Full sliding window |
| Optimizer | AdamW | AdamW | lr=1e-4 |
| Differential LR | graph_params lr × 0.1 | graph_params lr × 0.1 | Graph stability |
| Scheduler | Warmup(10) + CosineAnnealing | Warmup(10) + CosineAnnealing | — |
| Loss | ★ AW-MSE + λ·L1_sparse + λ_vq·VQ | **★ MSE + λ·L1_sparse + λ_vq·VQ (标准重建)** | 纯重建 MSE，取消异常加权 |
| **Channel Aggregation** | ★ Top-K (k=max(3, C×0.1)) | **★ Top-K (dim-adaptive, 3档比例)** | 分档自适应过滤噪声 |
| **Scale Aggregation** | ★ Cross-Scale Attention | Cross-Scale Attention | 可学习注意力加权 |
| λ_anomaly_weight | 10.0 (固定) | **None (占位符, 未使用)** | P1-A FIXED: 不在训练阶段使用 |
| target_anomaly_ratio | — | **0.20 (占位符, 未使用)** | P1-A FIXED: 不在训练阶段使用 |
| λ_vq | 0.1 | 0.1 | VQ commitment loss weight |
| vq_cooldown_epochs | 15 | 15 | Freeze non-VQ params initially |
| vq_codebook_size | 64 | 64 | Codebook entries |
| VQ mode | Dual-Path (旁路) | Dual-Path (旁路) | 主路径无损，VQ 仅评分 |
| **BoundaryDetector** | ★ recon=1.0,bd=0.3 | recon=1.0,bd=0.3 | 重建误差主评分 |
| **FocalLoss** | ★ REMOVED | REMOVED | 重建范式不适用 |
| **Top-K Peak Extraction** | ★ REMOVED | REMOVED | 硬截断破坏阈值 |
| **detect_topk_ratio** | ★ REMOVED | REMOVED | 默认超参移除 |
| dropout | 0.25 | 0.25 | — |
| batch_size | 256 | 256 | — |
| EarlyStopping | patience=15 | patience=15 | — |
| Max epochs | 100 | 100 | — |
| Mixed precision | bfloat16 | bfloat16 | (if supported) |
| warmup_epochs | 10 | 10 | — |
| lambda_causal_l1 | 0.001 | 0.001 | — |

### DDP (8-GPU)

```
distributed_worker_v10.py (subprocess via torchrun)
  - Same model, same config, same loss
  - per_gpu_batch=256 → effective batch = 2048
  - DistributedSampler for data partition
  - Only rank 0 saves checkpoint + runs evaluation
  - Back to main process → clean CUDA → next dataset
```

---

## Bug Fixes (v11.2 P0, 2026-05-19)

| Bug | Symptom | Root Cause | Fix |
|:---|:---|:---|:---|
| **[Bug 1]** VQ aggregation causal leakage | `mode='replicate'` padding copies future VQ distances | Wrong padding mode | Changed to `mode='constant', value=0` |
| **[Bug 2]** VQ distance not normalized | vq_dist (e-5~e-3) too small to affect anomaly score | No normalization | Added min-max normalization per batch |
| **[Bug 3]** Graph over-smoothing | Channel embedding homogenization | 2-layer MP | Reverted to single-layer + residual |
| **[Bug 4]** Dynamic scale degeneration | `[win//4, win//2, win]` dedup may yield <3 scales | Duplicate sizes after division | Added scale interpolation if dedup < 3 |
| **[Bug 5]** Shape alignment | `RuntimeError` when L < max(win_sizes) | score / vq_score / boundary_score 长度不一致 | 四处截断/填充对齐 |
| **★ P0 [Fix 1]** Top-K Channel Aggregation | 单通道 max 对噪声敏感 | max-pooling 选择最高噪声通道 | Top-K average 过滤噪声通道 |
| **★ P0 [Fix 2]** Attention Aggregation | 固定权重无法适应不同异常长度 | 全局统一加权 | Cross-Scale Attention + sigmoid 门控 |

---

## Evaluation Metrics

| Metric | Description | Expected on SWaT (v11.2 P0) |
|:---|:---|:---:|
| **affiliation_f** | Point-level boundary F1 (paper's Table II F1) | **0.82-0.88** (≥5%) |
| adjusted_f_score | Event-level with 7-point tolerance | >0.96 |
| **raw_f_score** (≥2%) | Exact point-wise reconstruction error | >0.30 |
| **raw_f_score** (1%) | Exact point-wise reconstruction error | ≥0.15 (期望) |
| aupr | Area under precision-recall curve | >0.90 |
| auc | Area under ROC curve | >0.95 |

---

## File Structure

```
ts_benchmark/baselines/self_impl/LaGraph/
├── LaGraph.py                  # ★ P0: 移除 FocalLoss + Top-K Peak Extraction + detect_topk_ratio
├── gcn_model.py                # ★ P0: Top-K Channel + Cross-Scale Attention + Dual-Path VQ + Weight Balance
├── graph_learner.py            # ChannelAdaptiveGraph + SimplifiedTemporalGraph
├── temporal_encoder.py         # BoundaryDetector + EncoderStack
├── attention.py                # OrdAttention
├── decomp.py                   # MoE series decomposition
├── RevIN.py                    # Reversible instance normalization
├── channel_mask.py             # Channel masking
├── vq_bottleneck.py            # VQ Bottleneck (Dual-Path, serial_mode=False)
├── distributed_worker_v10.py   # DDP training worker
├── distributed_worker.py       # (v9 legacy, deprecated)
├── graph_evolution.py          # (v6 legacy, deprecated)
└── vq_bottleneck.py            # (v7 legacy, deprecated)
```

---

## Architecture Evolution

```
v10:  Baseline with serial VQ + fixed multi-scale
   │
   ▼
v11.1: Bug fixes (causal leakage, normalization, over-smoothing, scale degeneration)
   │
   ▼
v11.2 (pre-P0): 四项根本改进
   ├── BoundaryDetector:        解决重建范式崩溃
   ├── Dynamic Scale Selection: 解决静态尺度缺陷
   ├── Dual-Path VQ:           解决VQ量化信息丢失
   └── Cross-Scale Attention:  解决多尺度固定权重问题
   │
   ▼
v11.2 P0 ★ (2026-05-19): P0 修正路线
   ├── ① Top-K Channel Aggregation    ✅ 替代 max(dim=-1)，过滤噪声通道
   ├── ② Cross-Scale Attention完整实现 ✅ 可学习 query + sigmoid 门控
   ├── ③ Dual-Path VQ (旁路模式)       ✅ serial_mode=False
   ├── ④ BoundaryDetector 权重平衡     ✅ recon=1.0, bd=0.3 (防 Hard Nuke)
   ├── ⑤ 移除 FocalLoss               ✅ 重建范式不适用
   ├── ⑥ 移除 Top-K Peak Extraction    ✅ 硬截断破坏阈值连续性
   └── ⑦ 移除 detect_topk_ratio参数    ✅ 默认超参清理
   │
   ▼
v11.3 P1-A FIXED ★ (2026-05-20): P1 修正反馈
   ├── ① P1-A 移除训练阶段异常加权      ✅ SegLoader 无真实标签，恢复纯 MSE
   ├── ② P1-C 低维通道自适应 Top-K     ✅ 按通道维度分档（≥50:10%, 20-50:15%, <20:25%）
   └── ③ ARCHITECTURE.md 文档同步     ✅ P1-A FIXED 完整记录
```

---

## Known Limitations (v11.3 P1-FIXED)

| 限制 | 表现 | 影响范围 | 可能的后续修复 |
|:---|:---|:---:|:---|
| 跨数据集差距 | SWaT vs MSL raw_f 差距 ≥ 0.10 | 泛化性 | 需要更通用的自适应机制 |
| 图模块负贡献 | MSL 上 GCN 可能引入噪声 | 简单单变量数据 | 通道级 gating |
| VQ codebook 稳定性 | 加权梯度可能影响 codebook 训练 | 训练早期 | 监控 VQ利用率+EMA更新 |
| 弱信号灵敏度 | Top-K 聚合在 C 很小时 k=3 可能仍含噪声 | 低维度数据 (C≤10) | 自适应 k = max(1, C×0.2) |
| P1-A 异常加权未在训练实现 | 仅在检测阶段通过 percentile 阈值替代 | 训练信号不足 | 若未来有真实标签可恢复 |
| BoundaryDetector 未参数化 | 固定 sigmoid 门控，无 ratio-aware 自适应 | 不同异常率切换 | Ratio-Adaptive Gate (未来修复) |

---

*Last updated: May 20, 2026 — v11.3 P1-FIXED SparseLaGraph (0.4M params, Standard MSE + L1 + VQ Loss, Dimension-Adaptive Top-K)*
