# LaGraph Changelog

---

## 2026-05-14 — v11 超参系统性修正（过拟合修复）

### 背景
通过对 `train_history.json` × 10+ 实验的系统分析，发现 v10 存在 4 个关键过拟合驱动参数和一个 LR 调度器问题。

### 修改汇总

| # | 参数 | 旧值 | 新值 | 影响 |
|---|------|------|------|------|
| 1 | `batch_size` | 2048-4096 | **256** | 降低梯度方差，提升泛化 |
| 2 | `dropout` | 0.1 | **0.25** | 增强正则化 |
| 3 | `weight_decay` | 1e-5 (硬编码) | **1e-4** | L2 约束增强 10x |
| 4 | `lambda_vq` | 0.01 | **0.1** | VQ 参与训练 (0.3%→3%) |
| 5 | `warmup_epochs` | 5 | **10** | codebook 稳定初始化 |
| 6 | `vq_cooldown_epochs` | 10 | **15** | 与 warmup 对齐 |
| 7 | `lambda_causal_l1` | 0.0001 | **0.001** | 图稀疏性增强 10x |
| 8 | `anomaly_ratio` | 9 个值 | **6 个值** | 减少冗余计算 |
| 9 | LR Scheduler | ReduceLROnPlateau | **CosineAnnealing** | 避免过拟合后无效降 LR |
| 10 | 多尺度窗口 | `[50,100,200]`→退化为2 | **动态3尺度** | 恢复3尺度融合 |

### 文件修改

**1. `ts_benchmark/baselines/self_impl/LaGraph/LaGraph.py`**
- `DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS`: batch_size=256, dropout=0.25, lambda_vq=0.1, 
  warmup_epochs=10, anomaly_ratio→6个值, lambda_causal_l1=0.001
- `_single_gpu_train`: weight_decay 从硬编码 1e-5 改为 `self.config.weight_decay`（默认 1e-4）
- `_single_gpu_train`: LR 调度从 dict（warmup+plateau）改为 `SequentialLR(warmup+CosineAnnealing)`

**2. `config/all_detect_label_config.json`** (已确认超参引用方式)
**3. `scripts/run_diagnostic_ablation.py`** — batch_size 2048→256

---

## 2026-05-14 — v11 Bugfix: DDP 冻结参数崩溃 + trained 标志顺序

## 2026-05-14 — v11 Bugfix: DDP 冻结参数崩溃 + trained 标志顺序

### 背景
v11 新增 VQ Bottleneck 冷却期机制：前 `vq_cooldown_epochs`（默认 10）轮冻结所有非 VQ 参数，让 codebook 先初始化正常模式的原型向量。该机制在单卡路径验证通过，但 DDP 多卡路径存在两个 Bug。

---

### Bug 1: DDP `find_unused_parameters` 未设置
**文件**: `ts_benchmark/baselines/self_impl/LaGraph/distributed_worker_v10.py`

**现象**: torchrun 启动后 OCC 通信报错，子进程崩溃。

**根因**: VQ 冷却期冻结非 VQ 参数（`requires_grad=False`），DDP reducer 在每次迭代后尝试同步所有参数的梯度，但冻结参数的梯度为 `None`。`find_unused_parameters=False`（默认）时 reducer 要求所有参数参与损失计算，冻结参数违反该契约。

**修复**: 设置 `find_unused_parameters=True`，允许 DDP reducer 跳过未参与计算的参数。

```python
model = DDP(model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=True,
            gradient_as_bucket_view=True)
```

---

### Bug 2: 冷却期损失计算未隔离
**文件**: `ts_benchmark/baselines/self_impl/LaGraph/distributed_worker_v10.py` + `LaGraph.py`

**现象**: 即便设置了 `find_unused_parameters=True`，冷却期内的损失仍然混合 MSE + VQ loss，冻结参数的 `grad_fn` 链虽被切断但不会崩溃，但训练行为不符合设计意图。

**根因**: 冷却期的目的是**仅更新 codebook**，但如果保留 MSE loss 项，所有非 VQ 参数也会被求导（只是 `requires_grad=False` 阻止了梯度更新），实际上制造了无意义的计算图。

**修复**: 冷却期内 `loss = aux_losses['vq_loss']`，完全排除 MSE 和其他损失项。

```python
# ★ v11 FIX: 冷却期内仅使用 VQ loss（其他参数冻结，MSE 无梯度流动）
if aux_losses and 'vq_loss' in aux_losses and epoch < vq_cooldown_epochs:
    loss = aux_losses['vq_loss']
else:
    loss = F.mse_loss(rec, input_data)
    if aux_losses and 'sparse_loss' in aux_losses:
        loss = loss + lambda_causal_l1 * aux_losses['sparse_loss']
    if aux_losses and 'vq_loss' in aux_losses:
        loss = loss + lambda_vq * aux_losses['vq_loss']
```

---

### Bug 3: `trained` 标志设置顺序错误
**文件**: `ts_benchmark/baselines/self_impl/LaGraph/LaGraph.py`

**现象**: DDP 训练完成后，`detect_fit` 调用 `self.detect_score(self._train_raw)` 时报错 `RuntimeError: Model not trained yet. Call detect_fit first.`

**根因**: `self.trained` 在 `detect_score` 调用完成后才设置，但 `detect_score` 方法第一行就检查 `self.trained`。

```python
# 旧代码
self._train_anomaly_scores, _ = self.detect_score(self._train_raw)  # ← 崩溃
self.trained = True
```

**修复**: 将 `self.trained = True` 移到 `detect_score` 调用之前。

```python
# 新代码
self.trained = True
self._train_anomaly_scores, _ = self.detect_score(self._train_raw)
```

---

### 经验教训

1. **DDP 动态冻结参数**必须设置 `find_unused_parameters=True`，且需要评估性能影响（会略慢于默认模式，因为 reducer 需要检测 unused params）。
2. **冷却期损失隔离**不仅仅是正确性问题——还影响模型行为：如果保留 MSE，冻结参数的梯度反向传播到 codebook 输入，干扰 codebook 学习。
3. **标志设置顺序**是经典 off-by-one bug：在调用依赖 `trained` 标志的方法之前必须先设置它。

### 关联文件
- `ts_benchmark/baselines/self_impl/LaGraph/distributed_worker_v10.py`
- `ts_benchmark/baselines/self_impl/LaGraph/LaGraph.py`

</write_to_file>
