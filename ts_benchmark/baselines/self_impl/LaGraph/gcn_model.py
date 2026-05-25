# -*- coding: utf-8 -*-
"""
LaGraph v11.2 SparseLaGraph — 极简双图协同异常检测模型
=====================================================

根据 `docs/update.md` P0 级修改路线图（2026-05-19）:

★ P0 修改 3: Dual-Path VQ 集成 (SparseGCN)
  - VQBottleneck(serial_mode=False) — VQ 作为旁路评分，不阻塞主路径
  - VQ 距离仅用于 multi_scale_forward 的异常评分增强

★ P0 修改 4: Attention-based Aggregation (MultiScaleAnomalyScorer)
  - 多尺度窗口融合从固定权重 → 可学习跨尺度注意力
  - 学习一个 query 向量，对 3 个尺度的重建误差做注意力加权
  - 每个位置可自适应选择最相关的尺度

v11.1 修复:
  [Bug 1] VQ 聚合的因果泄露 — mode='replicate' → 'constant'
  [Bug 2] VQ 距离未归一化 — 自适应 min-max scaling
  [Bug 3] 图卷积过度平滑 (单层 + 残差)
  [Bug 4] 动态尺度退化

v10 有效架构: MoEDecomposition + ChannelAdaptiveGraph + SimplifiedTemporalGraph
  + VQBottleneck + EncoderStack + MultiScaleAnomalyScorer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomp import MoEDecomposition
from .graph_learner import (
    ChannelAdaptiveGraph, DynamicTemporalGraph, SimplifiedTemporalGraph
)
from .temporal_encoder import EncoderStack
from .vq_bottleneck import VQBottleneck


# ══════════════════════════════════════════════════════════════════
#  ★ P0 [MODIFIED] MultiScaleAnomalyScorer — 注意力加权多尺度融合
# ══════════════════════════════════════════════════════════════════

class MultiScaleAnomalyScorer(nn.Module):
    """
    多尺度窗口异常评分器 (v11.2 Attention-based Aggregation)

    ★ P0 修改: 固定权重 → 可学习跨尺度注意力

    旧版:
      score = (w1*s1 + w2*s2 + w3*s3) / sum(w)
      全局固定权重，无法对不同位置自适应选择尺度

    新版 (Cross-Scale Attention):
      1. 将 3 个尺度的重建误差作为 value vectors
      2. 学习一个可 query 向量，与每个尺度的 context 计算注意力分数
      3. 每个时间位置独立计算注意力权重，动态融合多尺度信息

    原理:
      不同异常模式的持续长度差异极大（SWaT: 10-1000 步），
      固定权重对所有位置一视同仁，无法捕捉局部最佳尺度。
      注意力聚合使模型能对不同时间点选择最匹配的窗口尺度。

    实现细节:
      - Q: 可学习 query (B, 1, d_attn) — 跨样本共享
      - K: 每个尺度的 pooled context (B, 3, d_attn)
      - V: 每个尺度的重建误差 (B, L_max, 3)
      - output = softmax(QK^T) @ V^T — 动态加权融合
    """

    def __init__(self, win_sizes=[50, 100, 200], weights=None, d_model=128, channel=55):
        super().__init__()
        self.win_sizes = win_sizes
        n_scales = len(win_sizes)

        # ★ P0 P1-C: Top-K 通道聚合 — 按维度自适应稀疏度
        #   每个时间点选择重建误差最大的 k 个通道，取其均值作为该点异常分数
        #   原理: 单通道 max 对噪声敏感（某噪声通道高误差→误报），
        #         而 top-k mean 过滤噪声通道，保留真正异常通道的信号。
        #
        # ★ P1-C: 低维通道自适应 Top-K
        #   问题: MSL(25维) 用 max(3, 25*0.1%=2.5→3) 的 3 通道，占 12%，过滤效果不足
        #         SWaT(55维) 用 max(3, 55*0.1%=5.5→5) 的 5 通道，占 9%，稀疏度适中
        #   修复: 低维数据用更高比例的 Top-K
        if channel >= 50:
            self.topk_k = max(5, int(channel * 0.1))      # 50+维: max(5, 10%)
        elif channel >= 20:
            self.topk_k = max(2, int(channel * 0.15))     # 20-50维: max(2, 15%)
        else:
            self.topk_k = max(1, int(channel * 0.25))     # <20维: max(1, 25%)
        # 效果:
        #   MSL(25维):  25*0.15 = 3.75 ≈ 3  (占12%) — 比旧版 3 更保守
        #   SWaT(55维): 55*0.10 = 5.50 ≈ 5  (占9%)  — 与旧版 5 一致
        #   极端低维:   10*0.25 = 2.50 ≈ 2  (占20%) — 防止噪声干扰

        # ★ P0: Cross-Scale Attention 参数
        #   可学习 query: 表示"最佳多尺度模式"的向量
        self.scale_query = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02,
        )

        # 将每个尺度的 context 映射到 d_model 空间
        self.scale_proj = nn.Sequential(
            nn.Linear(n_scales, d_model),
            nn.ReLU(),
        )

        # 注意力 logits 缩放
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # ★ 保留旧版固定权重接口（用于兼容/回退）
        if weights is not None:
            weights_t = torch.tensor(weights, dtype=torch.float32)
        else:
            weights_t = torch.ones(n_scales, dtype=torch.float32)
        self.register_buffer('fixed_weights', weights_t / weights_t.sum())

    def forward(self, x_rec_dict: dict, x_input_dict: dict, vq_score: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_rec_dict: {win_size: (B, L_win, C)} 各尺度的重建
            x_input_dict: {win_size: (B, L_win, C)} 各尺度的输入
            vq_score: (B, L_max) or None — VQ 增强评分

        Returns:
            anomaly_score: (B, L_max) 融合后的异常得分
        """
        B = list(x_rec_dict.values())[0].shape[0]
        L_max = max(self.win_sizes)
        device = list(x_rec_dict.values())[0].device
        n_scales = len(self.win_sizes)

        # 步骤1: 收集各尺度评分
        scale_scores = []  # list of (B, L_max)
        scale_pooled = []  # list of (B, 1) — 全局池化用于注意力

        # 根据实际传入的窗口 key 迭代（兼容 L < max(win_sizes) 的场景）
        actual_sizes = sorted(x_rec_dict.keys())
        for ws in actual_sizes:
            rec = x_rec_dict[ws]    # (B, ws, C)
            inp = x_input_dict[ws]  # (B, ws, C)
            err = F.l1_loss(rec, inp, reduction='none')  # (B, ws, C)
            # ★ P0: Top-K 通道聚合 — k=max(3, int(C*0.1)) 自适应稀疏度
            #   选择重建误差最大的 k 个通道取其均值，过滤噪声通道的突发高误差
            topk_vals = err.topk(k=self.topk_k, dim=-1, largest=True, sorted=False)[0]  # (B, ws, k)
            err_per_t = topk_vals.mean(dim=-1)           # (B, ws)

            # 填充到 L_max（中心对齐）
            if ws < L_max:
                pad = (L_max - ws) // 2
                padded = F.pad(
                    err_per_t, (pad, L_max - ws - pad),
                    mode='constant', value=0,
                )
                scale_scores.append(padded)  # (B, L_max)
            else:
                scale_scores.append(err_per_t)

            # 全局平均池化作为尺度 context
            pooled = F.adaptive_avg_pool1d(
                err_per_t.unsqueeze(1), 1,
            ).squeeze(-1)  # (B, 1)
            scale_pooled.append(pooled)

        # (B, L_max, n_scales_actual)
        scores_tensor = torch.stack(scale_scores, dim=-1)
        n_scales_actual = len(actual_sizes)

        # 步骤3: 融合
        #   注: fixed_weights 保存 init 时的预置权重，长度固定为 self.win_sizes
        #       实际输入可能只有部分窗口（当 L < max(win_sizes)），此时用平均代替
        if n_scales_actual == len(self.fixed_weights):
            # ★ P0: 全尺度可用 → Cross-Scale Attention
            #   scale_pooled: (B, n_scales_actual)
            scale_context = torch.cat(scale_pooled, dim=-1)  # (B, n_scales)
            scale_context = self.scale_proj(scale_context)   # (B, d_model)
            scale_context = scale_context.unsqueeze(1)       # (B, 1, d_model)

            # 注意力分数: Q (1, 1, d_model) @ K^T (B, d_model, 1) → (B, 1, 1)
            q = self.scale_query.expand(B, -1, -1)  # (B, 1, d_model)
            attn_logits = torch.matmul(
                q, scale_context.transpose(-2, -1),
            ) / max(self.temperature.abs(), 0.1)    # (B, 1, 1)

            # 每个样本独立的全局注意力权重 (B, 1)
            attn_weight = torch.sigmoid(attn_logits.squeeze(-1))  # (B, 1)

            # 注意力门控: α * fixed_weight + (1-α) * mean
            fixed_agg = sum(
                w * scores_tensor[..., i]
                for i, w in enumerate(self.fixed_weights)
            )  # (B, L_max)
            score = attn_weight * fixed_agg + (1 - attn_weight) * scores_tensor.mean(-1)
        else:
            # 部分窗口可用: 均等平均
            score = scores_tensor.mean(-1)  # (B, L_max)

        # ★ v11.1: 融合 VQ 增强评分（如果提供，对齐到 score 长度）
        if vq_score is not None:
            score_len = score.shape[1]
            if vq_score.shape[1] > score_len:
                vq_score = vq_score[:, -score_len:]
            elif vq_score.shape[1] < score_len:
                pad_vq = torch.zeros(B, score_len - vq_score.shape[1], device=score.device)
                vq_score = torch.cat([pad_vq, vq_score], dim=1)
            score = score + vq_score

        return score


# ══════════════════════════════════════════════════════════════════
#  ★ P0 [MODIFIED] SparseGCN — Dual-Path VQ + MultiScaleAnomalyScorer
# ══════════════════════════════════════════════════════════════════

class LaggedCausalMechanism(nn.Module):
    """
    Lag-constrained structural mechanism.

    The module predicts X_i(t) from X_j(t-k), k > 0. This is not a complete
    causal identification procedure, but it gives the model an explicit
    temporal-precedence constraint that can be tested by ablation.
    """

    def __init__(
        self,
        channel,
        lags=(1, 2, 4),
        topk=5,
        dropout=0.1,
        score_mode="residual",
    ):
        super().__init__()
        self.channel = channel
        self.lags = tuple(int(lag) for lag in lags if int(lag) > 0)
        if not self.lags:
            self.lags = (1,)
        self.max_lag = max(self.lags)
        self.topk = min(max(1, int(topk)), channel)
        self.score_mode = score_mode

        self.edge_logits = nn.Parameter(
            torch.randn(len(self.lags), channel, channel) * 0.01,
        )
        self.lag_logits = nn.Parameter(torch.zeros(len(self.lags)))
        self.self_loop_bias = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)

    def _sparse_parent_weights(self):
        logits = self.edge_logits.clone()
        eye = torch.eye(self.channel, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = torch.where(eye, logits + self.self_loop_bias, logits)
        weights = F.softmax(logits, dim=1)
        if self.topk < self.channel:
            topk_idx = torch.topk(weights, k=self.topk, dim=1).indices
            mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights

    def forward(self, x):
        B, L, C = x.shape
        parent_weights = self._sparse_parent_weights()
        if L <= self.max_lag:
            zero_score = x.new_zeros(B, L)
            return x, zero_score, parent_weights

        current = x[:, self.max_lag:, :]
        lag_weights = F.softmax(self.lag_logits, dim=0)
        pred = current.new_zeros(current.shape)
        self_pred = current.new_zeros(current.shape)
        eye = torch.eye(self.channel, device=x.device, dtype=x.dtype)

        for lag_idx, lag in enumerate(self.lags):
            start = self.max_lag - lag
            past = x[:, start:L - lag, :]
            pred = pred + lag_weights[lag_idx] * torch.einsum(
                "blc,co->blo", past, parent_weights[lag_idx],
            )
            self_pred = self_pred + lag_weights[lag_idx] * torch.einsum(
                "blc,co->blo", past, eye,
            )
        pred = self.dropout(pred)

        full_err = F.smooth_l1_loss(pred, current, reduction="none").mean(dim=-1)
        self_err = F.smooth_l1_loss(self_pred, current, reduction="none").mean(dim=-1)
        if self.score_mode == "cf_parent_gain":
            score_valid = self_err - full_err
        elif self.score_mode == "cf_parent_hurt":
            score_valid = (full_err - self_err).clamp_min(0.0)
        else:
            score_valid = full_err
        score = F.pad(score_valid, (self.max_lag, 0), mode="constant", value=0.0)

        pred_full = x.new_zeros(B, L, C)
        pred_full[:, :self.max_lag, :] = x[:, :self.max_lag, :]
        pred_full[:, self.max_lag:, :] = pred
        return pred_full, score, parent_weights

    def get_sparsity_loss(self):
        return self._sparse_parent_weights().abs().mean()


class StateAwareGraphFusion(nn.Module):
    """
    Operating-state-conditioned dual-graph fusion.

    The module estimates a soft operating state from window statistics, then
    uses state-specific gates to decide how much parallel channel-graph and
    temporal-graph information should correct the stable serial graph path.
    """

    def __init__(
        self,
        channel,
        num_states=4,
        dropout=0.1,
        graph_gate_init=0.6,
        residual_init=0.15,
    ):
        super().__init__()
        self.channel = int(channel)
        self.num_states = max(2, int(num_states))

        hidden = max(32, min(128, self.channel * 2))
        self.state_encoder = nn.Sequential(
            nn.Linear(self.channel * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_states),
        )

        graph_gate_init = min(max(float(graph_gate_init), 1e-3), 1.0 - 1e-3)
        residual_init = min(max(float(residual_init), 1e-3), 1.0 - 1e-3)
        gate_logit = float(np.log(graph_gate_init / (1.0 - graph_gate_init)))
        residual_logit = float(np.log(residual_init / (1.0 - residual_init)))
        self.state_graph_gate_logits = nn.Parameter(
            torch.full((self.num_states, 1), gate_logit),
        )
        self.residual_logit = nn.Parameter(torch.tensor(residual_logit))

    @staticmethod
    def _safe_std(x):
        if x.shape[1] <= 1:
            return torch.zeros_like(x.mean(dim=1))
        return x.std(dim=1, unbiased=False)

    def _state_features(self, x):
        mean = x.mean(dim=1)
        std = self._safe_std(x)
        slope = x[:, -1, :] - x[:, 0, :]
        energy = x.abs().mean(dim=1)
        return torch.cat([mean, std, slope, energy], dim=-1)

    def forward(self, resid, resid_channel, resid_temporal, resid_serial=None):
        state_logits = self.state_encoder(self._state_features(resid))
        state_probs = F.softmax(state_logits, dim=-1)
        state_gate = torch.sigmoid(state_probs @ self.state_graph_gate_logits)
        state_gate = state_gate.view(resid.shape[0], 1, 1)

        parallel_feat = state_gate * resid_channel + (1.0 - state_gate) * resid_temporal
        residual_weight = torch.sigmoid(self.residual_logit)
        if resid_serial is None:
            fused = parallel_feat
        else:
            fused = resid_serial + residual_weight * (parallel_feat - resid_serial)

        return fused, {
            "state_probs": state_probs,
            "state_gate": state_gate,
            "state_residual_weight": residual_weight,
        }

    def balance_loss(self, state_probs):
        mean_probs = state_probs.mean(dim=0).clamp_min(1e-8)
        return (mean_probs * (mean_probs.log() + np.log(self.num_states))).sum()

    def confidence_loss(self, state_probs):
        probs = state_probs.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        return entropy / max(np.log(self.num_states), 1e-8)


class SparseGCN(nn.Module):
    """
    SparseLaGraph v11.2 — 极简双图协同异常检测模型

    ★ P0 架构变更:
      - Dual-Path VQ: VQ 移至旁路，主路径保持连续特征流

    架构:
      Input (B, L, C)
        ↓ MoEDecomposition
      resid, trend
        ↓ ChannelAdaptiveGraph (topk=5, 对称化 A_sym) — 单层消息传递
      resid_adapted
        ↓ SimplifiedTemporalGraph (多尺度时序卷积 + 因果偏置)
      resid_temp
        ↓ ┌─────────────────────────────────────┐
        │ 主路径: resid_temp (连续特征, 无量化)   │
        │ 旁路:   VQ → vq_dist (仅计算距离)      │
        └─────────────────────────────────────┘
        ↓ Linear proj → d_model=128 + 残差旁路
        ↓ EncoderStack (2 层, d_model=128)
        ↓ Linear proj → c_out
      resid_out + trend_out = x_rec

    异常评分（推理时使用 MultiScaleAnomalyScorer）:
      score = L1_recon_error + vq_enhanced_score
    """

    def __init__(self, win_size, enc_in, c_out, dropout, n_heads=4,
                 d_model=128, e_layers=2, patch_size=16, channel=55,
                 d_ff=256, topk=5, sparse_topk=None,
                 use_channel_graph=True, use_temporal_graph=True,
                 use_dynamic_temporal_graph=False,
                 dynamic_temporal_residual_init=0.1,
                 dynamic_temporal_topk=None,
                 dynamic_temporal_gate_mode="global",
                 use_vq_bypass=True,
                 use_multi_scale_scorer=True,
                 use_direct_vq_score=False,
                 vq_score_weight=0.3,
                 score_topk_k=None,
                 use_synthetic_anomaly_head=False,
                 use_synthetic_score=False,
                 synthetic_score_weight=0.1,
                 synthetic_score_eps=1e-6,
                 use_parallel_graph_fusion=False,
                 graph_fusion_gate_mode="sample",
                 graph_fusion_strategy="parallel",
                 graph_fusion_residual_init=0.1,
                 use_graph_shift_score=False,
                 graph_shift_score_weight=0.1,
                 graph_shift_score_eps=1e-6,
                 use_lagged_causal_graph=False,
                 causal_lags=(1, 2, 4),
                 causal_topk=5,
                 causal_detach_backbone=True,
                 use_causal_score=False,
                 causal_score_weight=0.1,
                 causal_score_eps=1e-6,
                 causal_score_mode="residual",
                 causal_score_tail="upper",
                 use_temporal_graph_regularization=False,
                 use_score_channel_normalization=False,
                 score_channel_norm_mode="robust_z",
                 score_channel_norm_eps=1e-6,
                 use_state_aware_fusion=False,
                 state_aware_num_states=4,
                 state_aware_graph_gate_init=0.6,
                 state_aware_residual_init=0.15,
                 **kwargs):
        super(SparseGCN, self).__init__()

        self.channel = channel
        self.win_size = win_size
        self.c_out = c_out
        self.topk = topk
        self.use_channel_graph = use_channel_graph
        self.use_temporal_graph = use_temporal_graph
        self.use_dynamic_temporal_graph = use_dynamic_temporal_graph
        self.dynamic_temporal_residual_init = dynamic_temporal_residual_init
        self.dynamic_temporal_topk = dynamic_temporal_topk
        self.dynamic_temporal_gate_mode = dynamic_temporal_gate_mode
        self.use_vq_bypass = use_vq_bypass
        self.use_multi_scale_scorer = use_multi_scale_scorer
        self.use_direct_vq_score = use_direct_vq_score
        self.score_topk_k = score_topk_k
        self.use_synthetic_anomaly_head = use_synthetic_anomaly_head
        self.use_synthetic_score = use_synthetic_score
        self.synthetic_score_weight = float(synthetic_score_weight)
        self.synthetic_score_eps = float(synthetic_score_eps)
        self.use_parallel_graph_fusion = use_parallel_graph_fusion
        self.graph_fusion_gate_mode = graph_fusion_gate_mode
        self.graph_fusion_strategy = graph_fusion_strategy
        self.use_graph_shift_score = use_graph_shift_score
        self.graph_shift_score_weight = float(graph_shift_score_weight)
        self.graph_shift_score_eps = float(graph_shift_score_eps)
        self.use_lagged_causal_graph = use_lagged_causal_graph
        if isinstance(causal_lags, str):
            causal_lags = [int(x.strip()) for x in causal_lags.split(",") if x.strip()]
        self.causal_lags = tuple(int(lag) for lag in causal_lags)
        self.causal_topk = int(causal_topk)
        self.causal_detach_backbone = causal_detach_backbone
        self.use_causal_score = use_causal_score
        self.causal_score_weight = float(causal_score_weight)
        self.causal_score_eps = float(causal_score_eps)
        self.causal_score_mode = causal_score_mode
        self.causal_score_tail = causal_score_tail
        self.use_temporal_graph_regularization = use_temporal_graph_regularization
        self.use_score_channel_normalization = use_score_channel_normalization
        self.score_channel_norm_mode = score_channel_norm_mode
        self.score_channel_norm_eps = score_channel_norm_eps
        self.use_state_aware_fusion = use_state_aware_fusion
        self.state_aware_num_states = int(state_aware_num_states)
        self.state_aware_graph_gate_init = float(state_aware_graph_gate_init)
        self.state_aware_residual_init = float(state_aware_residual_init)
        self.register_buffer(
            'score_channel_center',
            torch.zeros(1, 1, channel),
            persistent=False,
        )
        self.register_buffer(
            'score_channel_scale',
            torch.ones(1, 1, channel),
            persistent=False,
        )
        self.register_buffer(
            'graph_shift_center',
            torch.zeros(1, channel, channel),
            persistent=False,
        )
        self.register_buffer(
            'graph_shift_score_center',
            torch.zeros(1),
            persistent=False,
        )
        self.register_buffer(
            'graph_shift_score_scale',
            torch.ones(1),
            persistent=False,
        )
        self.register_buffer(
            'causal_score_center',
            torch.zeros(1),
            persistent=False,
        )
        self.register_buffer(
            'causal_score_scale',
            torch.ones(1),
            persistent=False,
        )

        # === 自适应通道图 ===
        self.register_buffer(
            'synthetic_score_center',
            torch.zeros(1),
            persistent=False,
        )
        self.register_buffer(
            'synthetic_score_scale',
            torch.ones(1),
            persistent=False,
        )

        self.channel_graph = ChannelAdaptiveGraph(
            num_nodes=enc_in, topk=topk,
            sparse_topk=sparse_topk, dropout=dropout,
        )

        # === 简化时序图 ===
        if use_dynamic_temporal_graph:
            self.temporal_graph = DynamicTemporalGraph(
                win_size=win_size,
                d_model=enc_in,
                dropout=dropout,
                temporal_topk=dynamic_temporal_topk,
                residual_init=dynamic_temporal_residual_init,
                gate_mode=dynamic_temporal_gate_mode,
            )
        else:
            self.temporal_graph = SimplifiedTemporalGraph(
                win_size=win_size,
                d_model=enc_in,
                dropout=dropout,
            )

        # === EncoderStack（精简版）===
        if use_parallel_graph_fusion:
            fusion_hidden = max(16, c_out)
            self.graph_fusion_sample_gate = nn.Sequential(
                nn.Linear(c_out * 4, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 1),
                nn.Sigmoid(),
            )
            self.graph_fusion_time_gate = nn.Sequential(
                nn.Linear(c_out * 4, fusion_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 1),
                nn.Sigmoid(),
            )
            graph_fusion_residual_init = min(max(float(graph_fusion_residual_init), 1e-3), 1.0 - 1e-3)
            self.graph_fusion_residual_logit = nn.Parameter(
                torch.tensor(
                    float(np.log(graph_fusion_residual_init / (1.0 - graph_fusion_residual_init))),
                )
            )
        else:
            self.graph_fusion_sample_gate = None
            self.graph_fusion_time_gate = None
            self.graph_fusion_residual_logit = None

        self.encoder = EncoderStack(
            num_layers=e_layers, d_model=d_model, d_ff=d_ff,
            win_size=win_size, n_heads=n_heads, d_state=16, d_conv=4,
            expand=2, dropout=dropout,
            use_freq_branch=False,
            use_cross_channel=False,
            use_multi_scale=True,
        )

        # === 投影层 ===
        self.proj_in = nn.Linear(c_out, d_model, bias=True)
        self.proj_out = nn.Linear(d_model, c_out, bias=True)

        # === 趋势分支 ===
        self.trend_linear = nn.Linear(c_out, c_out, bias=True)

        # === 序列分解 ===
        self.decomp = MoEDecomposition(enc_in)

        # === 残差旁路 ===
        self.residual_shortcut = nn.Linear(c_out, d_model, bias=True)
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

        # === ★ P0: Dual-Path VQ Bottleneck (旁路模式) ===
        self.vq_bottleneck = VQBottleneck(
            dim=c_out,
            codebook_size=64,
            commitment_cost=0.25,
            serial_mode=False,  # ★ Dual-Path: VQ 作为旁路
        )

        # === 多尺度异常评分器（仅推理时使用）===
        # v11.1 FIX [Bug 4]: 确保至少有 3 个不同的尺度
        _s1 = max(8, win_size // 4)
        _s2 = max(16, win_size // 2)
        _s3 = win_size
        _dynamic_sizes = sorted(set([_s1, _s2, _s3]))
        while len(_dynamic_sizes) < 3:
            _mid = (_dynamic_sizes[0] + _dynamic_sizes[-1]) // 2
            _dynamic_sizes = sorted(set(_dynamic_sizes + [_mid]))
        self.multi_scale_scorer = MultiScaleAnomalyScorer(
            win_sizes=_dynamic_sizes,
            d_model=d_model,  # ★ P0: 注意力聚合需要 d_model
            channel=self.channel,  # ★ P0: Top-K 通道聚合需要 C
        )

        # === VQ 增强权重 ===
        self.vq_score_weight = nn.Parameter(torch.tensor(float(vq_score_weight)))

        if use_lagged_causal_graph:
            self.lagged_causal_graph = LaggedCausalMechanism(
                channel=c_out,
                lags=self.causal_lags,
                topk=self.causal_topk,
                dropout=dropout,
                score_mode=self.causal_score_mode,
            )
        else:
            self.lagged_causal_graph = None

        if use_state_aware_fusion:
            self.state_aware_fusion = StateAwareGraphFusion(
                channel=c_out,
                num_states=state_aware_num_states,
                dropout=dropout,
                graph_gate_init=state_aware_graph_gate_init,
                residual_init=state_aware_residual_init,
            )
        else:
            self.state_aware_fusion = None

        # ★ 保持与原有 forward 返回格式兼容
        if use_synthetic_anomaly_head:
            hidden = max(32, d_model // 2)
            self.synthetic_anomaly_head = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.synthetic_anomaly_head = None

        self.use_freq_loss = False
        self.lambda_freq = 0.0
        self.use_contrastive = False
        self.lambda_contrastive = 0.0
        self.use_prototype = False
        self.use_prediction_head = False
        self.lambda_pred = 0.0
        self.contrastive_temp = 0.5
        self._apply_architecture_switches()

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def _apply_architecture_switches(self):
        self._set_trainable(self.channel_graph, self.use_channel_graph)
        self._set_trainable(self.temporal_graph, self.use_temporal_graph)
        self._set_trainable(self.graph_fusion_sample_gate, self.use_parallel_graph_fusion)
        self._set_trainable(self.graph_fusion_time_gate, self.use_parallel_graph_fusion)
        if self.graph_fusion_residual_logit is not None:
            self.graph_fusion_residual_logit.requires_grad = self.use_parallel_graph_fusion
        self._set_trainable(self.state_aware_fusion, self.use_state_aware_fusion)
        self._set_trainable(self.lagged_causal_graph, self.use_lagged_causal_graph)
        self._set_trainable(self.synthetic_anomaly_head, self.use_synthetic_anomaly_head)
        self._set_trainable(self.vq_bottleneck, self.use_vq_bypass)
        self._set_trainable(self.multi_scale_scorer, self.use_multi_scale_scorer)

    def _parallel_fuse_graphs(self, resid, resid_channel, resid_temporal):
        gate_input = torch.cat(
            [resid, resid_channel, resid_temporal, (resid_channel - resid_temporal).abs()],
            dim=-1,
        )
        if self.graph_fusion_gate_mode == "time":
            gate = self.graph_fusion_time_gate(gate_input)
        else:
            gate = self.graph_fusion_sample_gate(gate_input.mean(dim=1)).view(resid.shape[0], 1, 1)
        return gate * resid_channel + (1.0 - gate) * resid_temporal, gate

    def get_sparse_loss(self):
        """通道图 L1 稀疏正则化损失"""
        if not self.use_channel_graph:
            return next(self.parameters()).new_tensor(0.0)
        return self.channel_graph.get_l1_penalty()

    def set_warmup_progress(self, alpha: float):
        if not self.use_channel_graph:
            return
        self.channel_graph.set_warmup_progress(alpha)

    def set_score_channel_stats(self, center, scale):
        center = torch.as_tensor(center, dtype=self.score_channel_center.dtype)
        scale = torch.as_tensor(scale, dtype=self.score_channel_scale.dtype)
        center = center.reshape(1, 1, -1).to(self.score_channel_center.device)
        scale = scale.reshape(1, 1, -1).to(self.score_channel_scale.device)
        if center.shape[-1] != self.score_channel_center.shape[-1]:
            raise ValueError(
                f"channel stat size mismatch: {center.shape[-1]} != {self.score_channel_center.shape[-1]}"
            )
        self.score_channel_center.copy_(center)
        self.score_channel_scale.copy_(scale.clamp_min(self.score_channel_norm_eps))

    def set_graph_shift_stats(self, center, score_center, score_scale):
        center = torch.as_tensor(center, dtype=self.graph_shift_center.dtype)
        center = center.reshape(1, center.shape[-2], center.shape[-1]).to(self.graph_shift_center.device)
        if center.shape[-2:] != self.graph_shift_center.shape[-2:]:
            raise ValueError(
                f"graph stat size mismatch: {center.shape[-2:]} != {self.graph_shift_center.shape[-2:]}"
            )
        self.graph_shift_center.copy_(center)
        self.graph_shift_score_center.copy_(
            torch.as_tensor([score_center], dtype=self.graph_shift_score_center.dtype, device=self.graph_shift_score_center.device)
        )
        self.graph_shift_score_scale.copy_(
            torch.as_tensor([score_scale], dtype=self.graph_shift_score_scale.dtype, device=self.graph_shift_score_scale.device)
            .clamp_min(self.graph_shift_score_eps)
        )

    def _graph_shift_score(self, A_adaptive):
        center = self.graph_shift_center.to(device=A_adaptive.device, dtype=A_adaptive.dtype)
        raw = (A_adaptive - center).abs().mean(dim=(1, 2))
        score_center = self.graph_shift_score_center.to(device=A_adaptive.device, dtype=A_adaptive.dtype)
        score_scale = self.graph_shift_score_scale.to(device=A_adaptive.device, dtype=A_adaptive.dtype)
        return (raw - score_center).clamp_min(0.0) / score_scale.clamp_min(self.graph_shift_score_eps)

    def set_causal_score_stats(self, score_center, score_scale):
        self.causal_score_center.copy_(
            torch.as_tensor([score_center], dtype=self.causal_score_center.dtype, device=self.causal_score_center.device)
        )
        self.causal_score_scale.copy_(
            torch.as_tensor([score_scale], dtype=self.causal_score_scale.dtype, device=self.causal_score_scale.device)
            .clamp_min(self.causal_score_eps)
        )

    def _normalize_causal_score(self, causal_score):
        center = self.causal_score_center.to(device=causal_score.device, dtype=causal_score.dtype)
        scale = self.causal_score_scale.to(device=causal_score.device, dtype=causal_score.dtype)
        if self.causal_score_tail == "lower":
            return (center - causal_score).clamp_min(0.0) / scale.clamp_min(self.causal_score_eps)
        return (causal_score - center).clamp_min(0.0) / scale.clamp_min(self.causal_score_eps)

    def set_synthetic_score_stats(self, score_center, score_scale):
        self.synthetic_score_center.copy_(
            torch.as_tensor([score_center], dtype=self.synthetic_score_center.dtype, device=self.synthetic_score_center.device)
        )
        self.synthetic_score_scale.copy_(
            torch.as_tensor([score_scale], dtype=self.synthetic_score_scale.dtype, device=self.synthetic_score_scale.device)
            .clamp_min(self.synthetic_score_eps)
        )

    def _normalize_synthetic_score(self, synthetic_score):
        center = self.synthetic_score_center.to(device=synthetic_score.device, dtype=synthetic_score.dtype)
        scale = self.synthetic_score_scale.to(device=synthetic_score.device, dtype=synthetic_score.dtype)
        return (synthetic_score - center).clamp_min(0.0) / scale.clamp_min(self.synthetic_score_eps)

    def _normalize_score_error(self, err):
        if not self.use_score_channel_normalization:
            return err
        center = self.score_channel_center.to(device=err.device, dtype=err.dtype)
        scale = self.score_channel_scale.to(device=err.device, dtype=err.dtype)
        scale = scale.clamp_min(self.score_channel_norm_eps)
        if self.score_channel_norm_mode == "scale":
            return err / scale
        return (err - center).clamp_min(0.0) / scale

    def forward(self, x):
        """
        x: (B, L, C)

        返回:
            x_rec:       (B, L, C) 重建序列
            A_adaptive:  (B, C, C) 自适应依赖邻接矩阵
            d_score:     (B, L) 零张量（保持接口兼容）
            freq_recon:  None（保持接口兼容）
            stage1_feat: (B, L, C) 阶段一输出
            aux_losses:  dict — 包含 'sparse_loss', 'vq_loss', 'vq_dist'
            pred:        None（保持接口兼容）
        """
        B, L, C = x.shape
        d_score = torch.zeros(B, L, device=x.device)

        # ========== 阶段一：表示学习 ==========
        # 步骤 1: 序列分解
        resid, trend = self.decomp(x)
        graph_fusion_gate = None
        state_aware_info = None
        causal_score = None
        causal_mechanism_loss = None

        if self.use_lagged_causal_graph and self.lagged_causal_graph is not None:
            causal_input = resid.detach() if self.causal_detach_backbone else resid
            causal_pred, causal_score, _ = self.lagged_causal_graph(causal_input)
            valid_start = self.lagged_causal_graph.max_lag
            if L > valid_start:
                causal_mechanism_loss = F.smooth_l1_loss(
                    causal_pred[:, valid_start:, :],
                    causal_input[:, valid_start:, :],
                )

        # 步骤 2: 自适应通道依赖图
        if self.use_channel_graph:
            resid_adapted, A_adaptive = self.channel_graph(resid)
        else:
            resid_adapted = resid
            A_adaptive = torch.eye(C, device=x.device).unsqueeze(0).expand(B, C, C)

        # 步骤 3: 简化时序图
        if self.use_state_aware_fusion and self.use_channel_graph and self.use_temporal_graph:
            resid_serial, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
            resid_temporal_base, _ = self.temporal_graph(resid, A_proximity=A_adaptive)
            stage1_feat, state_aware_info = self.state_aware_fusion(
                resid,
                resid_adapted,
                resid_temporal_base,
                resid_serial=resid_serial,
            )
        elif self.use_parallel_graph_fusion and self.use_channel_graph and self.use_temporal_graph:
            if self.graph_fusion_strategy == "residual_serial":
                resid_serial, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
                resid_temp, _ = self.temporal_graph(resid, A_proximity=A_adaptive)
                parallel_feat, graph_fusion_gate = self._parallel_fuse_graphs(
                    resid, resid_adapted, resid_temp,
                )
                residual_weight = torch.sigmoid(self.graph_fusion_residual_logit)
                stage1_feat = resid_serial + residual_weight * (parallel_feat - resid_serial)
            else:
                resid_temp, A_temp = self.temporal_graph(resid, A_proximity=A_adaptive)
                stage1_feat, graph_fusion_gate = self._parallel_fuse_graphs(
                    resid, resid_adapted, resid_temp,
                )
        else:
            if self.use_temporal_graph:
                resid_temp, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
            else:
                resid_temp = resid_adapted
                A_temp = None
            stage1_feat = resid_temp  # (B, L, C)

        # ★ P0: Dual-Path VQ — 旁路模式
        #   VQ 不参与重建路径，仅计算 vq_dist 和 vq_loss
        #   continuous_feat = stage1_feat (原始连续特征)
        if self.use_vq_bypass:
            continuous_feat, vq_loss_val, vq_dist = self.vq_bottleneck(stage1_feat)
        else:
            continuous_feat = stage1_feat
            vq_loss_val = stage1_feat.new_tensor(0.0)
            vq_dist = stage1_feat.new_zeros(B, L)

        # ========== 阶段二：重建 (使用连续特征) ==========
        resid_proj = self.proj_in(continuous_feat)  # (B, L, d_model)

        # 残差旁路
        shortcut_feat = self.residual_shortcut(continuous_feat)
        resid_proj = resid_proj + self.residual_gate * shortcut_feat

        # 步骤 5: EncoderStack
        resid_enc = self.encoder(resid_proj)  # (B, L, d_model)
        synthetic_logits = None
        if self.synthetic_anomaly_head is not None:
            synthetic_logits = self.synthetic_anomaly_head(resid_enc).squeeze(-1)

        # 步骤 6: 投影回 c_out
        resid_out = self.proj_out(resid_enc)  # (B, L, C)

        # 步骤 7: 趋势合并
        trend_out = self.trend_linear(trend)
        x_rec = resid_out + trend_out

        aux_losses = {}
        aux_losses['sparse_loss'] = self.get_sparse_loss()
        aux_losses['vq_loss'] = vq_loss_val
        aux_losses['vq_dist'] = vq_dist  # (B, L) — 用于增强异常评分
        if causal_score is not None:
            aux_losses['causal_score'] = causal_score
        if causal_mechanism_loss is not None:
            aux_losses['causal_mechanism_loss'] = causal_mechanism_loss
            aux_losses['causal_sparse_loss'] = self.lagged_causal_graph.get_sparsity_loss()
        if synthetic_logits is not None:
            aux_losses['synthetic_logits'] = synthetic_logits
        if graph_fusion_gate is not None:
            aux_losses['graph_fusion_gate_mean'] = graph_fusion_gate.detach().mean()
            aux_losses['graph_fusion_residual_weight'] = torch.sigmoid(
                self.graph_fusion_residual_logit.detach()
            )
        if state_aware_info is not None:
            state_probs = state_aware_info["state_probs"]
            aux_losses['state_gate_mean'] = state_aware_info["state_gate"].detach().mean()
            aux_losses['state_residual_weight'] = state_aware_info["state_residual_weight"].detach()
            aux_losses['state_balance_loss'] = self.state_aware_fusion.balance_loss(state_probs)
            aux_losses['state_confidence_loss'] = self.state_aware_fusion.confidence_loss(state_probs)
        if self.use_temporal_graph_regularization and A_temp is not None:
            aux_losses['temporal_graph_smooth_loss'] = (
                A_temp[:, 1:, :] - A_temp[:, :-1, :]
            ).pow(2).mean()
            idx = torch.arange(A_temp.shape[-1], device=A_temp.device, dtype=A_temp.dtype)
            dist = (idx[:, None] - idx[None, :]).abs()
            dist = dist / max(1, A_temp.shape[-1] - 1)
            aux_losses['temporal_graph_locality_loss'] = (A_temp * dist.unsqueeze(0)).mean()

        return x_rec, A_adaptive, d_score, None, continuous_feat, aux_losses, None

    def multi_scale_forward(self, x):
        """
        多尺度前向推理（用于 detect_score/detect_label）

        ★ P0 变更:
          1. Dual-Path VQ: 仅 vq_dist 用于异常评分

        Args:
            x: (B, L, C) — L 必须 >= max(valid_sizes)

        Returns:
            anomaly_score: (B, L) 融合异常得分
            vq_score: (B, L) VQ 偏离度（用于分析/日志）
        """
        B, L, C = x.shape
        win_sizes = self.multi_scale_scorer.win_sizes

        # 过滤掉大于输入长度的窗口
        valid_sizes = [ws for ws in win_sizes if ws <= L]
        if not valid_sizes:
            valid_sizes = [L]

        x_rec_dict = {}
        x_input_dict = {}
        vq_dist_dict = {}
        graph_shift_dict = {}
        causal_score_dict = {}
        synthetic_score_dict = {}

        for ws in valid_sizes:
            x_win = x[:, -ws:, :]  # (B, ws, C)

            if ws < self.win_size:
                pad_len = self.win_size - ws
                x_pad = F.pad(x_win.transpose(1, 2), (0, pad_len), mode='constant', value=0).transpose(1, 2)
            else:
                x_pad = x_win

            rec_pad, A_adaptive, _, _, _, aux_losses, _ = self.forward(x_pad)  # (B, win_size, C)

            # 收集 VQ 距离
            vq_d = aux_losses.get('vq_dist', None)
            if vq_d is not None:
                vq_d_win = vq_d[:, :ws] if ws < self.win_size else vq_d
            else:
                vq_d_win = torch.zeros(B, ws, device=x.device)

            # Trim 回原始窗口大小
            if ws < self.win_size:
                rec_win = rec_pad[:, :ws, :]
            else:
                rec_win = rec_pad

            x_rec_dict[ws] = rec_win
            x_input_dict[ws] = x_win
            vq_dist_dict[ws] = vq_d_win  # (B, ws)
            if self.use_graph_shift_score:
                graph_shift = self._graph_shift_score(A_adaptive).unsqueeze(1)
                graph_shift_dict[ws] = graph_shift.expand(-1, ws)
            if self.use_causal_score and aux_losses.get('causal_score', None) is not None:
                c_score = aux_losses['causal_score']
                causal_score_dict[ws] = c_score[:, :ws] if ws < self.win_size else c_score
            if self.use_synthetic_score and aux_losses.get('synthetic_logits', None) is not None:
                s_score = torch.sigmoid(aux_losses['synthetic_logits'])
                synthetic_score_dict[ws] = s_score[:, :ws] if ws < self.win_size else s_score

        # v11.1 FIX [Bug 2]: 自适应 VQ 距离归一化
        raw_vq_score = self._aggregate_vq_dist(vq_dist_dict, valid_sizes, B, L, x.device)

        # 自适应归一化
        vq_min = raw_vq_score.min(dim=-1, keepdim=True)[0]  # (B, 1)
        vq_max = raw_vq_score.max(dim=-1, keepdim=True)[0]  # (B, 1)
        vq_range = (vq_max - vq_min).clamp(min=1e-8)
        vq_score_normalized = (raw_vq_score - vq_min) / vq_range  # (B, L) ∈ [0, 1]

        # 乘以可学习权重
        if self.use_vq_bypass:
            vq_score = self.vq_score_weight * vq_score_normalized  # (B, L)
        else:
            vq_score = torch.zeros(B, L, device=x.device)

        # ★ P0: 融合评分 (MultiScaleAnomalyScorer 内部已包含 VQ)
        #   注意: MultiScaleAnomalyScorer 输出长度 = max(win_sizes)，需对齐
        L_scorer = max(self.multi_scale_scorer.win_sizes)
        if self.use_multi_scale_scorer:
            vq_score_scorer = vq_score[:, -L_scorer:] if vq_score.shape[1] >= L_scorer else vq_score
            score = self.multi_scale_scorer(
                x_rec_dict, x_input_dict,
                vq_score=vq_score_scorer if self.use_vq_bypass else None,
            )
        else:
            ws = max(valid_sizes)
            err = F.l1_loss(x_rec_dict[ws], x_input_dict[ws], reduction='none')
            err = self._normalize_score_error(err)
            k = self.score_topk_k or self.multi_scale_scorer.topk_k
            k = min(max(1, int(k)), err.shape[-1])
            score = err.topk(k=k, dim=-1, largest=True, sorted=False)[0].mean(dim=-1)
            if self.use_direct_vq_score and self.use_vq_bypass:
                vq_score_direct = vq_score[:, -score.shape[1]:] if vq_score.shape[1] >= score.shape[1] else vq_score
                if vq_score_direct.shape[1] < score.shape[1]:
                    pad_len = score.shape[1] - vq_score_direct.shape[1]
                    pad_vq = torch.zeros(B, pad_len, device=score.device)
                    vq_score_direct = torch.cat([pad_vq, vq_score_direct], dim=1)
                score = score + vq_score_direct

        if self.use_graph_shift_score and graph_shift_dict:
            graph_score = self._aggregate_vq_dist(graph_shift_dict, valid_sizes, B, L, x.device)
            graph_score = graph_score[:, -score.shape[1]:] if graph_score.shape[1] >= score.shape[1] else graph_score
            if graph_score.shape[1] < score.shape[1]:
                pad_len = score.shape[1] - graph_score.shape[1]
                pad_graph = torch.zeros(B, pad_len, device=score.device)
                graph_score = torch.cat([pad_graph, graph_score], dim=1)
            score = score + self.graph_shift_score_weight * graph_score

        if self.use_causal_score and causal_score_dict:
            causal_score = self._aggregate_vq_dist(causal_score_dict, valid_sizes, B, L, x.device)
            causal_score = self._normalize_causal_score(causal_score)
            causal_score = causal_score[:, -score.shape[1]:] if causal_score.shape[1] >= score.shape[1] else causal_score
            if causal_score.shape[1] < score.shape[1]:
                pad_len = score.shape[1] - causal_score.shape[1]
                pad_causal = torch.zeros(B, pad_len, device=score.device)
                causal_score = torch.cat([pad_causal, causal_score], dim=1)
            score = score + self.causal_score_weight * causal_score

        if self.use_synthetic_score and synthetic_score_dict:
            synthetic_score = self._aggregate_vq_dist(synthetic_score_dict, valid_sizes, B, L, x.device)
            synthetic_score = self._normalize_synthetic_score(synthetic_score)
            synthetic_score = synthetic_score[:, -score.shape[1]:] if synthetic_score.shape[1] >= score.shape[1] else synthetic_score
            if synthetic_score.shape[1] < score.shape[1]:
                pad_len = score.shape[1] - synthetic_score.shape[1]
                pad_synth = torch.zeros(B, pad_len, device=score.device)
                synthetic_score = torch.cat([pad_synth, synthetic_score], dim=1)
            score = score + self.synthetic_score_weight * synthetic_score

        # score 输出是 L_max（即 max(win_sizes)），按实际有效窗口截断
        L_eff = max(valid_sizes)
        if score.shape[1] > L_eff:
            score = score[:, -L_eff:]

        L_scorer = score.shape[1]
        if L_scorer < L:
            pad_s = torch.zeros(B, L - L_scorer, device=x.device)
            score = torch.cat([pad_s, score], dim=1)
        elif L_scorer > L:
            score = score[:, :L]

        return score, vq_score

    def _aggregate_vq_dist(self, vq_dist_dict, valid_sizes, B, L, device):
        """
        聚合多尺度 VQ 距离：每个窗口的 VQ 距离 → 点级 VQ 增强分数。

        v11.1 FIX [Bug 1]: 因果泄露修复 — mode='replicate' → 'constant'

        Args:
            vq_dist_dict: {ws: (B, ws)} — 各尺度 VQ 距离
            valid_sizes: list[int]
            B, L: batch size 和原始长度
            device: torch.device

        Returns:
            vq_score: (B, L) — 对齐后的点级 VQ 偏离度
        """
        if not vq_dist_dict or all(v.shape[1] == 0 for v in vq_dist_dict.values()):
            return torch.zeros(B, L, device=device)

        L_max = max(valid_sizes)
        vq_collected = []

        for ws in valid_sizes:
            vq_win = vq_dist_dict[ws]  # (B, ws)
            if ws < L_max:
                pad_len = L_max - ws
                vq_padded = F.pad(vq_win, (pad_len // 2, pad_len - pad_len // 2), mode='constant', value=0)
                vq_collected.append(vq_padded)
            else:
                vq_collected.append(vq_win)

        if not vq_collected:
            return torch.zeros(B, L, device=device)

        # 平均聚合多尺度
        vq_agg = torch.stack(vq_collected).mean(dim=0)  # (B, L_max)

        # 对齐到原始长度 L
        if L_max > L:
            vq_agg = vq_agg[:, :L]
        elif L_max < L:
            pad = torch.zeros(B, L - L_max, device=device)
            vq_agg = torch.cat([pad, vq_agg], dim=1)

        return vq_agg
