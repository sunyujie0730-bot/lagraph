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

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decomp import MoEDecomposition
from .graph_learner import (
    ChannelAdaptiveGraph,
    DirectedSparseChannelGraph,
    DynamicTemporalGraph,
    SimplifiedTemporalGraph,
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
            return x, zero_score, parent_weights, x

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
        self_pred_full = x.new_zeros(B, L, C)
        self_pred_full[:, :self.max_lag, :] = x[:, :self.max_lag, :]
        self_pred_full[:, self.max_lag:, :] = self_pred
        return pred_full, score, parent_weights, self_pred_full

    def get_sparsity_loss(self):
        return self._sparse_parent_weights().abs().mean()


class StrictCrossLagMechanism(nn.Module):
    """Separate self dynamics from delayed cross-channel explanations.

    ``self_pred`` only reads a target channel's own history.  ``full_pred``
    adds a sparse, strictly off-diagonal transfer term that reads other
    channels at positive lags.  This separation makes the prediction gain a
    meaningful response signal instead of another reconstruction residual.
    """

    def __init__(
        self,
        channel,
        lags=(1, 3, 6, 12),
        topk=5,
        dropout=0.0,
        use_static_prior=False,
    ):
        super().__init__()
        self.channel = int(channel)
        self.lags = tuple(int(lag) for lag in lags if int(lag) > 0) or (1,)
        self.max_lag = max(self.lags)
        self.topk = min(max(1, int(topk)), max(1, self.channel - 1))
        self.use_static_prior = bool(use_static_prior)

        n_lags = len(self.lags)
        self.self_lag_logits = nn.Parameter(torch.zeros(self.channel, n_lags))
        self.cross_edge_logits = nn.Parameter(
            torch.randn(n_lags, self.channel, self.channel) * 0.01
        )
        # Signed transfer allows an upstream deviation to suppress or amplify
        # a target channel while attention still selects only a few parents.
        self.cross_transfer_raw = nn.Parameter(
            torch.randn(n_lags, self.channel, self.channel) * 0.02
        )
        self.parent_activation_gain = nn.Parameter(torch.zeros(n_lags, self.channel))
        self.parent_activation_bias = nn.Parameter(torch.zeros(n_lags, self.channel))
        self.prior_strength_raw = nn.Parameter(torch.tensor(-4.0))
        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer("static_prior", torch.empty(0), persistent=True)

    def set_static_prior(self, prior):
        prior = torch.as_tensor(prior, dtype=self.cross_edge_logits.dtype)
        if prior.shape != (self.channel, self.channel):
            raise ValueError(
                "strict cross prior size mismatch: "
                f"expected {(self.channel, self.channel)}, got {tuple(prior.shape)}"
            )
        prior = prior.clamp_min(0.0)
        prior.fill_diagonal_(0.0)
        self.static_prior = prior.detach().clone()

    def _cross_attention(self):
        logits = self.cross_edge_logits
        if self.use_static_prior and self.static_prior.numel() > 0:
            prior = self.static_prior.to(device=logits.device, dtype=logits.dtype)
            prior_bias = torch.log(prior.clamp_min(1e-6)).unsqueeze(0)
            logits = logits + torch.sigmoid(self.prior_strength_raw) * prior_bias

        eye = torch.eye(self.channel, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        attention = F.softmax(logits, dim=1)
        if self.topk < self.channel - 1:
            topk_idx = torch.topk(attention, k=self.topk, dim=1).indices
            mask = torch.zeros_like(attention).scatter_(1, topk_idx, 1.0)
            attention = attention * mask
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attention

    def forward(self, x):
        B, L, C = x.shape
        if C != self.channel:
            raise ValueError(f"strict cross mechanism expected {self.channel} channels, got {C}")

        attention = self._cross_attention()
        edge_weight = attention * torch.tanh(self.cross_transfer_raw)
        if L <= self.max_lag:
            return x, x, attention, edge_weight

        current = x[:, self.max_lag:, :]
        self_pred = current.new_zeros(current.shape)
        cross_pred = current.new_zeros(current.shape)
        self_lag_weights = F.softmax(self.self_lag_logits, dim=-1)

        for lag_idx, lag in enumerate(self.lags):
            start = self.max_lag - lag
            past = x[:, start:L - lag, :]
            self_pred = self_pred + past * self_lag_weights[:, lag_idx].view(1, 1, C)

            # The activation is channel-local and only reads the parent's past.
            activation = 1.0 + 0.5 * torch.tanh(
                past.abs() * self.parent_activation_gain[lag_idx].view(1, 1, C)
                + self.parent_activation_bias[lag_idx].view(1, 1, C)
            )
            cross_pred = cross_pred + torch.einsum(
                "btc,co->bto",
                past * activation,
                edge_weight[lag_idx],
            )

        full_valid = self_pred + self.dropout(cross_pred)
        full_pred = x.new_zeros(B, L, C)
        full_pred[:, :self.max_lag, :] = x[:, :self.max_lag, :]
        full_pred[:, self.max_lag:, :] = full_valid
        self_pred_full = x.new_zeros(B, L, C)
        self_pred_full[:, :self.max_lag, :] = x[:, :self.max_lag, :]
        self_pred_full[:, self.max_lag:, :] = self_pred
        return full_pred, self_pred_full, attention, edge_weight

    def get_sparsity_loss(self):
        attention = self._cross_attention().clamp_min(1e-8)
        entropy = -(attention * attention.log()).sum(dim=1).mean()
        transfer = torch.tanh(self.cross_transfer_raw).abs().mean()
        return entropy + transfer


class SPSRoleHead(nn.Module):
    """Joint source-responsibility and response-role predictor for SPS."""

    def __init__(self, feature_dim=4, hidden=16, dropout=0.1):
        super().__init__()
        hidden = max(8, min(int(hidden), 64))
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source = nn.Linear(hidden, 1)
        self.response = nn.Linear(hidden, 1)
        nn.init.constant_(self.source.bias, -2.0)
        nn.init.constant_(self.response.bias, -2.0)

    def forward(self, features):
        hidden = self.trunk(features)
        return self.source(hidden).squeeze(-1), self.response(hidden).squeeze(-1)


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


class RootResponseRCAHead(nn.Module):
    """Root-response decomposition head for variable-level RCA.

    The head keeps a structural sign constraint: source evidence increases a
    variable's RCA logit, while graph-supported response evidence decreases it.
    This turns the diagnostic prior into an inductive bias instead of a
    post-hoc weighted score.
    """

    def __init__(
        self,
        source_dim,
        response_dim,
        hidden,
        dropout=0.1,
        response_penalty_init=1.0,
        source_confidence_discount=0.75,
    ):
        super().__init__()
        hidden = max(8, min(int(hidden), 16))
        response_hidden = max(4, min(hidden // 2, 8))
        self.source_confidence_discount = min(
            max(float(source_confidence_discount), 0.0),
            0.95,
        )
        self.source_net = nn.Sequential(
            nn.Linear(source_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.response_net = nn.Sequential(
            nn.Linear(response_dim, response_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(response_hidden, 1),
        )
        init = max(float(response_penalty_init), 1e-4)
        self.response_penalty_raw = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(init)))),
        )
        nn.init.constant_(self.source_net[-1].bias, -2.0)
        nn.init.constant_(self.response_net[-1].bias, 0.0)

    def forward(self, source_features, response_features):
        source_evidence = self.source_net(source_features).squeeze(-1)
        response_evidence = F.softplus(self.response_net(response_features).squeeze(-1))
        source_confidence = torch.sigmoid(source_evidence)
        response_discount = 1.0 - self.source_confidence_discount * source_confidence
        effective_response_evidence = response_discount * response_evidence
        response_penalty = F.softplus(self.response_penalty_raw)
        logits = source_evidence - response_penalty * effective_response_evidence
        return (
            logits,
            source_evidence,
            response_evidence,
            source_confidence,
            effective_response_evidence,
        )


class PairwiseRootResponseRCAHead(nn.Module):
    """Pairwise source-response head for variable-level RCA.

    The head learns a compact source-to-response relation matrix and uses it to
    reward variables that explain downstream response evidence while penalizing
    variables that are better explained as responses.
    """

    def __init__(
        self,
        source_dim,
        response_dim,
        hidden,
        channel,
        relation_rank=8,
        dropout=0.1,
        response_penalty_init=1.0,
        source_confidence_discount=0.75,
        graph_weight_init=0.75,
        explanation_reward_init=0.25,
        response_pair_penalty_init=0.50,
        pairwise_logit_weight=1.0,
    ):
        super().__init__()
        hidden = max(8, min(int(hidden), 16))
        response_hidden = max(4, min(hidden // 2, 8))
        self.channel = int(channel)
        self.relation_rank = max(2, int(relation_rank))
        self.pairwise_logit_weight = max(float(pairwise_logit_weight), 0.0)
        self.source_confidence_discount = min(
            max(float(source_confidence_discount), 0.0),
            0.95,
        )

        self.source_net = nn.Sequential(
            nn.Linear(source_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.response_net = nn.Sequential(
            nn.Linear(response_dim, response_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(response_hidden, 1),
        )
        self.source_relation = nn.Parameter(
            torch.empty(self.channel, self.relation_rank),
        )
        self.response_relation = nn.Parameter(
            torch.empty(self.channel, self.relation_rank),
        )
        self.relation_scale_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))

        response_penalty_init = max(float(response_penalty_init), 1e-4)
        explanation_reward_init = max(float(explanation_reward_init), 1e-4)
        response_pair_penalty_init = max(float(response_pair_penalty_init), 1e-4)
        graph_weight_init = min(max(float(graph_weight_init), 1e-3), 1.0 - 1e-3)
        self.response_penalty_raw = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(response_penalty_init)))),
        )
        self.explanation_reward_raw = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(explanation_reward_init)))),
        )
        self.response_pair_penalty_raw = nn.Parameter(
            torch.tensor(float(np.log(np.expm1(response_pair_penalty_init)))),
        )
        self.graph_weight_logit = nn.Parameter(
            torch.tensor(float(np.log(graph_weight_init / (1.0 - graph_weight_init)))),
        )

        nn.init.normal_(self.source_relation, mean=0.0, std=0.02)
        nn.init.normal_(self.response_relation, mean=0.0, std=0.02)
        nn.init.constant_(self.source_net[-1].bias, -2.0)
        nn.init.constant_(self.response_net[-1].bias, 0.0)

    def _relation_prior(self, A_adaptive, dtype, device):
        C = self.channel
        eye = torch.eye(C, device=device, dtype=dtype)
        learned_logits = (
            self.source_relation.to(dtype=dtype)
            @ self.response_relation.to(dtype=dtype).transpose(0, 1)
        ) / np.sqrt(float(self.relation_rank))
        learned_logits = F.softplus(self.relation_scale_raw).to(dtype=dtype) * learned_logits
        learned_logits = learned_logits.masked_fill(eye.bool(), -20.0)

        if A_adaptive is None:
            return torch.softmax(learned_logits, dim=-1).unsqueeze(0)

        graph = A_adaptive.to(dtype=dtype).clamp_min(0.0)
        graph = graph * (1.0 - eye.unsqueeze(0))
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        graph_logits = torch.log(graph.clamp_min(1e-8))
        graph_weight = torch.sigmoid(self.graph_weight_logit).to(dtype=dtype)
        logits = graph_weight * graph_logits + (1.0 - graph_weight) * learned_logits.unsqueeze(0)
        logits = logits.masked_fill(eye.bool().unsqueeze(0), -30.0)
        return torch.softmax(logits, dim=-1)

    def forward(self, source_features, response_features, A_adaptive=None):
        source_evidence = self.source_net(source_features).squeeze(-1)
        response_evidence = F.softplus(self.response_net(response_features).squeeze(-1))
        source_confidence = torch.sigmoid(source_evidence)
        relation = self._relation_prior(
            A_adaptive,
            dtype=source_features.dtype,
            device=source_features.device,
        )
        if relation.shape[0] == 1 and source_features.shape[0] > 1:
            relation = relation.expand(source_features.shape[0], -1, -1)

        source_support = torch.einsum("bij,blj->bli", relation, response_evidence)
        source_signal = source_confidence * F.softplus(source_evidence)
        response_support = torch.einsum("bij,bli->blj", relation, source_signal)
        response_discount = 1.0 - self.source_confidence_discount * source_confidence
        base_response_evidence = response_discount * response_evidence
        pair_response_evidence = F.softplus(self.response_pair_penalty_raw) * response_support
        effective_response_evidence = (
            base_response_evidence
            + self.pairwise_logit_weight * pair_response_evidence
        )
        logits = (
            source_evidence
            + self.pairwise_logit_weight
            * F.softplus(self.explanation_reward_raw)
            * source_support
            - F.softplus(self.response_penalty_raw) * effective_response_evidence
        )
        return (
            logits,
            source_evidence,
            response_evidence,
            source_confidence,
            effective_response_evidence,
            relation,
            source_support,
            response_support,
        )


class EventResponsibilityHead(nn.Module):
    """Point-wise channel responsibility head for event-level RCA."""

    def __init__(self, feature_dim, hidden=16, dropout=0.1):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class EventRouteHead(nn.Module):
    """Event-level gate between graph-source and event-responsibility evidence."""

    def __init__(self, feature_dim, hidden=16, dropout=0.1, init_alpha=0.55):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        init_alpha = min(max(float(init_alpha), 1e-3), 1.0 - 1e-3)
        init_bias = float(np.log(init_alpha / (1.0 - init_alpha)))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, features):
        logits = self.net(features).squeeze(-1)
        return logits, torch.sigmoid(logits)


class DualExpertFusionGate(nn.Module):
    """Select graph-conditioned or graph-independent temporal features per time step."""

    def __init__(self, feature_dim=5, hidden=16, dropout=0.1, init_graph_weight=0.5):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        init_graph_weight = min(max(float(init_graph_weight), 1e-3), 1.0 - 1e-3)
        init_bias = float(np.log(init_graph_weight / (1.0 - init_graph_weight)))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, features):
        return torch.sigmoid(self.net(features))


class SourceExpertHead(nn.Module):
    """Map fused temporal expert features to a source-variable logit."""

    def __init__(self, feature_dim=5, hidden=16, dropout=0.1):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class EventExpertGate(nn.Module):
    """Infer graph usefulness from the temporal evolution of an event window."""

    def __init__(self, feature_dim=5, hidden=16, dropout=0.1, init_graph_weight=0.5):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        init_graph_weight = min(max(float(init_graph_weight), 1e-3), 1.0 - 1e-3)
        init_bias = float(np.log(init_graph_weight / (1.0 - init_graph_weight)))
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, init_bias)

    def forward(self, features):
        states = self.temporal(features.transpose(1, 2))
        pooled = torch.cat([states.mean(dim=-1), states.amax(dim=-1)], dim=-1)
        return torch.sigmoid(self.output(pooled)).view(features.shape[0], 1, 1)


class ResponseSuppressorHead(nn.Module):
    """Predict whether a channel is graph-explained response evidence."""

    def __init__(self, feature_dim, hidden=16, dropout=0.1):
        super().__init__()
        hidden = max(8, min(int(hidden), 32))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features):
        return self.net(features).squeeze(-1)


class EvidenceFusionHead(nn.Module):
    """Learn evidence weights for variable-level RCA.

    Each variable is represented by a small vector of diagnostic evidence
    channels. The head predicts a simplex weight over those channels and uses
    the weighted evidence as the root-cause logit. This keeps the learned
    fusion interpretable and avoids a large variable-token Transformer before
    we have enough real RCA labels.
    """

    def __init__(self, evidence_dim, hidden=16, dropout=0.1, eps=1e-6):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.eps = float(eps)
        hidden = max(8, min(int(hidden), 32))
        self.gate_net = nn.Sequential(
            nn.Linear(self.evidence_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.evidence_dim),
        )
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.logit_bias = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, evidence):
        center = evidence.mean(dim=2, keepdim=True)
        scale = evidence.std(dim=2, keepdim=True, unbiased=False).clamp_min(self.eps)
        normalized = (evidence - center) / scale
        weights = torch.softmax(self.gate_net(normalized), dim=-1)
        fused = (weights * normalized).sum(dim=-1)
        logits = F.softplus(self.logit_scale_raw) * fused + self.logit_bias
        return logits, weights


class SourceInteractionHead(nn.Module):
    """Constrained source-interaction head for variable-level RCA.

    The head keeps the current RCA inductive bias differentiable: a source
    should have high local anomaly evidence and at least one source-support
    cue, rather than only a large propagated response.
    """

    def __init__(self, evidence_dim=5, eps=1e-6):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.eps = float(eps)
        if self.evidence_dim < 2:
            raise ValueError("SourceInteractionHead requires at least two evidence channels.")
        self.support_logits = nn.Parameter(torch.zeros(self.evidence_dim - 1))
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, evidence):
        positive = torch.log1p(torch.clamp_min(evidence, 0.0))
        denom = positive.amax(dim=2, keepdim=True).clamp_min(self.eps)
        normalized = positive / denom
        anchor = normalized[..., 0]
        support_weights = torch.softmax(self.support_logits, dim=0)
        support = (normalized[..., 1:] * support_weights).sum(dim=-1)
        interaction = torch.sqrt(torch.clamp_min(anchor * support, 0.0) + self.eps)
        logits = F.softplus(self.logit_scale_raw) * interaction + self.logit_bias
        return logits, support_weights


class SourceConsistencyHead(nn.Module):
    """Constrained source-consistency head for variable-level RCA.

    The head learns how strongly source-gate, onset, mechanism residual, and
    local anomaly evidence should agree before a variable is treated as a
    source. A graph-response feature is learned as a penalty so propagated
    responses are less likely to dominate the learned source score.
    """

    def __init__(self, evidence_dim=6, eps=1e-6):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.eps = float(eps)
        if self.evidence_dim < 6:
            raise ValueError("SourceConsistencyHead requires at least six evidence channels.")
        self.source_weight_logits = nn.Parameter(torch.zeros(5))
        self.response_penalty_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.5)))))
        self.agreement_mix_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.5)))))
        self.logit_scale_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, evidence):
        positive = torch.log1p(torch.clamp_min(evidence, 0.0))
        denom = positive.amax(dim=2, keepdim=True).clamp_min(self.eps)
        normalized = positive / denom

        source_evidence = torch.cat([normalized[..., :4], normalized[..., 5:6]], dim=-1)
        weights = torch.softmax(self.source_weight_logits, dim=0)
        additive = (source_evidence * weights).sum(dim=-1)
        geometric = torch.exp(
            (torch.log(source_evidence.clamp_min(self.eps)) * weights).sum(dim=-1)
        )
        agreement_mix = torch.sigmoid(self.agreement_mix_raw)
        source_support = (1.0 - agreement_mix) * additive + agreement_mix * geometric

        response_penalty = F.softplus(self.response_penalty_raw)
        response_evidence = normalized[..., 4]
        logits = (
            F.softplus(self.logit_scale_raw)
            * (source_support - response_penalty * response_evidence)
            + self.logit_bias
        )
        return logits, weights, source_support


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
                 use_channel_graph=True, channel_graph_evidence_only=False,
                 channel_graph_type="adaptive_symmetric",
                 directed_graph_parent_topk=5,
                 directed_graph_embedding_dim=8,
                 directed_graph_use_signed_transfer=False,
                 directed_graph_transfer_mode="low_rank",
                 directed_graph_transfer_rank=8,
                 directed_graph_transfer_scale=2.0,
                 use_channel_graph_reliability=False,
                 channel_graph_role="shared",
                 use_multilag_graph_propagation=False,
                 graph_propagation_lags=(0, 1, 2, 4, 8),
                 graph_propagation_lag_mode="static",
                 graph_propagation_alignment_temperature=0.2,
                 use_temporal_graph=True,
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
                 use_synthetic_rca_head=False,
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
                  use_causal_response_evidence=False,
                  use_strict_cross_mechanism=False,
                  strict_cross_lags=(1, 3, 6, 12),
                  strict_cross_topk=5,
                  strict_cross_detach_backbone=True,
                  strict_cross_use_channel_prior=False,
                  use_sps_role_head=False,
                  sps_role_hidden=16,
                  sps_role_detach_features=False,
                  use_temporal_graph_regularization=False,
                 use_score_channel_normalization=False,
                 score_channel_norm_mode="robust_z",
                 score_channel_norm_eps=1e-6,
                 use_channel_corr_prior=False,
                 channel_corr_prior_weight=0.0,
                 channel_corr_prior_bias=0.0,
                 lambda_channel_prior_align=0.0,
                 lambda_channel_mechanism=0.0,
                 use_channel_mechanism_score=False,
                 channel_mechanism_score_weight=0.1,
                 channel_mechanism_score_eps=1e-6,
                 use_mechanism_coupled_decoder=False,
                 mechanism_coupling_init=0.15,
                 use_source_preserving_decoder=False,
                 source_preserving_init=0.65,
                 source_preserving_detach_gate=True,
                 use_mechanism_residual_feedback=False,
                 mechanism_feedback_init=0.10,
                 mechanism_feedback_detach=True,
                 mechanism_feedback_norm="sample_l1",
                 mechanism_feedback_clip=3.0,
                 use_mechanism_predictive_head=False,
                 mechanism_predictive_blend_init=0.30,
                 use_source_gate=False,
                 source_gate_init=0.20,
                 use_onset_aware_source_gate=False,
                 source_gate_onset_window=8,
                 source_gate_onset_weight=0.5,
                 use_dual_expert_fusion=False,
                 dual_expert_hidden=16,
                 dual_expert_graph_init=0.5,
                 dual_expert_detach_inputs=True,
                 use_source_expert_branch=False,
                 source_expert_hidden=16,
                 source_expert_graph_init=0.5,
                 source_expert_detach_gate_inputs=True,
                 source_expert_gradient_checkpoint=False,
                 source_expert_gate_scope="time",
                 use_root_score_head=False,
                 root_score_head_mode="mlp",
                 root_score_detach_features=True,
                 root_response_penalty_init=1.0,
                 root_response_confidence_discount=0.75,
                 root_response_use_innovation_split=False,
                 use_pairwise_root_response_head=False,
                 pairwise_root_response_rank=8,
                 pairwise_root_response_graph_weight=0.75,
                 pairwise_root_response_reward_init=0.25,
                 pairwise_root_response_penalty_init=0.50,
                 pairwise_root_response_logit_weight=1.0,
                 use_event_responsibility_head=False,
                 event_responsibility_hidden=16,
                 event_responsibility_detach_features=False,
                 use_event_route_head=False,
                 event_route_hidden=16,
                 event_route_init=0.55,
                 event_route_detach_inputs=True,
                 event_route_response_penalty=0.0,
                 use_response_suppressor_head=False,
                 response_suppressor_hidden=16,
                 response_suppressor_detach_inputs=True,
                 use_evidence_fusion_head=False,
                 evidence_fusion_hidden=16,
                 evidence_fusion_detach_inputs=False,
                 evidence_fusion_use_graph_response=False,
                 use_source_interaction_head=False,
                 source_interaction_detach_inputs=True,
                 use_source_consistency_head=False,
                 source_consistency_detach_inputs=True,
                 use_channel_temporal_corefinement=False,
                 corefinement_init=0.10,
                 corefinement_detach_first_pass=True,
                 use_source_aware_corefinement=False,
                 source_aware_corefinement_init=0.15,
                 source_aware_corefinement_detach_gate=True,
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
        self.channel_graph_evidence_only = bool(channel_graph_evidence_only)
        self.channel_graph_type = str(channel_graph_type or "adaptive_symmetric").lower()
        self.directed_graph_parent_topk = int(directed_graph_parent_topk)
        self.directed_graph_embedding_dim = int(directed_graph_embedding_dim)
        self.directed_graph_use_signed_transfer = bool(directed_graph_use_signed_transfer)
        self.directed_graph_transfer_mode = str(
            directed_graph_transfer_mode or "low_rank"
        ).lower()
        self.directed_graph_transfer_rank = int(directed_graph_transfer_rank)
        self.directed_graph_transfer_scale = float(directed_graph_transfer_scale)
        self.use_channel_graph_reliability = bool(use_channel_graph_reliability)
        self.channel_graph_role = str(channel_graph_role or "shared").lower()
        if self.channel_graph_role not in {"shared", "response_only"}:
            raise ValueError(f"Unsupported channel_graph_role={self.channel_graph_role!r}")
        self.use_multilag_graph_propagation = bool(use_multilag_graph_propagation)
        if isinstance(graph_propagation_lags, str):
            graph_propagation_lags = [
                int(v.strip()) for v in graph_propagation_lags.split(",") if v.strip()
            ]
        self.graph_propagation_lags = tuple(
            sorted({max(0, int(v)) for v in graph_propagation_lags})
        ) or (0,)
        self.graph_propagation_lag_mode = str(
            graph_propagation_lag_mode or "static"
        ).lower()
        if self.graph_propagation_lag_mode not in {"static", "alignment"}:
            raise ValueError(
                "Unsupported graph_propagation_lag_mode="
                f"{self.graph_propagation_lag_mode!r}"
            )
        if self.use_multilag_graph_propagation:
            if self.graph_propagation_lag_mode == "static":
                self.graph_propagation_lag_logits = nn.Parameter(
                    torch.zeros(channel, len(self.graph_propagation_lags))
                )
                self.register_parameter("graph_propagation_alignment_temp_raw", None)
            else:
                self.register_parameter("graph_propagation_lag_logits", None)
                initial_temp = max(
                    float(graph_propagation_alignment_temperature),
                    1e-3,
                )
                self.graph_propagation_alignment_temp_raw = nn.Parameter(
                    torch.tensor(float(math.log(math.expm1(initial_temp))))
                )
        else:
            self.register_parameter("graph_propagation_lag_logits", None)
            self.register_parameter("graph_propagation_alignment_temp_raw", None)
        self._last_graph_propagation_lag_weights = None
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
        self.use_synthetic_rca_head = bool(use_synthetic_rca_head)
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
        self.use_causal_response_evidence = bool(use_causal_response_evidence)
        self.use_strict_cross_mechanism = bool(use_strict_cross_mechanism)
        if isinstance(strict_cross_lags, str):
            strict_cross_lags = [
                int(value.strip()) for value in strict_cross_lags.split(",") if value.strip()
            ]
        self.strict_cross_lags = tuple(int(lag) for lag in strict_cross_lags)
        self.strict_cross_topk = int(strict_cross_topk)
        self.strict_cross_detach_backbone = bool(strict_cross_detach_backbone)
        self.strict_cross_use_channel_prior = bool(strict_cross_use_channel_prior)
        self.use_sps_role_head = bool(use_sps_role_head)
        self.sps_role_hidden = int(sps_role_hidden)
        self.sps_role_detach_features = bool(sps_role_detach_features)
        self.use_temporal_graph_regularization = use_temporal_graph_regularization
        self.use_score_channel_normalization = use_score_channel_normalization
        self.score_channel_norm_mode = score_channel_norm_mode
        self.score_channel_norm_eps = score_channel_norm_eps
        self.use_channel_corr_prior = bool(use_channel_corr_prior)
        self.channel_corr_prior_weight = float(channel_corr_prior_weight)
        self.channel_corr_prior_bias = float(channel_corr_prior_bias)
        self.lambda_channel_prior_align = float(lambda_channel_prior_align)
        self.lambda_channel_mechanism = float(lambda_channel_mechanism)
        self.use_channel_mechanism_score = bool(use_channel_mechanism_score)
        self.channel_mechanism_score_weight = float(channel_mechanism_score_weight)
        self.channel_mechanism_score_eps = float(channel_mechanism_score_eps)
        self.use_mechanism_coupled_decoder = bool(use_mechanism_coupled_decoder)
        self.mechanism_coupling_init = float(mechanism_coupling_init)
        self.use_source_preserving_decoder = bool(use_source_preserving_decoder)
        self.source_preserving_init = float(source_preserving_init)
        self.source_preserving_detach_gate = bool(source_preserving_detach_gate)
        self.use_mechanism_residual_feedback = bool(use_mechanism_residual_feedback)
        self.mechanism_feedback_init = float(mechanism_feedback_init)
        self.mechanism_feedback_detach = bool(mechanism_feedback_detach)
        self.mechanism_feedback_norm = str(mechanism_feedback_norm or "sample_l1").lower()
        self.mechanism_feedback_clip = float(mechanism_feedback_clip)
        self.use_mechanism_predictive_head = bool(use_mechanism_predictive_head)
        self.mechanism_predictive_blend_init = float(mechanism_predictive_blend_init)
        self.use_source_gate = bool(use_source_gate)
        self.source_gate_init = float(source_gate_init)
        self.use_onset_aware_source_gate = bool(use_onset_aware_source_gate)
        self.source_gate_onset_window = max(1, int(source_gate_onset_window))
        self.source_gate_onset_weight = max(0.0, float(source_gate_onset_weight))
        self.use_dual_expert_fusion = bool(use_dual_expert_fusion)
        self.dual_expert_hidden = int(dual_expert_hidden)
        self.dual_expert_graph_init = float(dual_expert_graph_init)
        self.dual_expert_detach_inputs = bool(dual_expert_detach_inputs)
        self.use_source_expert_branch = bool(use_source_expert_branch)
        self.source_expert_hidden = int(source_expert_hidden)
        self.source_expert_graph_init = float(source_expert_graph_init)
        self.source_expert_detach_gate_inputs = bool(
            source_expert_detach_gate_inputs
        )
        self.source_expert_gradient_checkpoint = bool(
            source_expert_gradient_checkpoint
        )
        self.source_expert_gate_scope = str(
            source_expert_gate_scope or "time"
        ).lower()
        if self.source_expert_gate_scope not in {"time", "event"}:
            raise ValueError(
                "Unsupported source_expert_gate_scope="
                f"{self.source_expert_gate_scope!r}"
            )
        self.use_root_score_head = bool(use_root_score_head)
        self.root_score_head_mode = str(root_score_head_mode or "mlp").lower()
        self.root_score_detach_features = bool(root_score_detach_features)
        self.root_response_penalty_init = float(root_response_penalty_init)
        self.root_response_confidence_discount = min(
            max(float(root_response_confidence_discount), 0.0),
            0.95,
        )
        self.root_response_use_innovation_split = bool(root_response_use_innovation_split)
        self.use_pairwise_root_response_head = bool(use_pairwise_root_response_head)
        self.pairwise_root_response_rank = int(pairwise_root_response_rank)
        self.pairwise_root_response_graph_weight = float(pairwise_root_response_graph_weight)
        self.pairwise_root_response_reward_init = float(pairwise_root_response_reward_init)
        self.pairwise_root_response_penalty_init = float(pairwise_root_response_penalty_init)
        self.pairwise_root_response_logit_weight = float(pairwise_root_response_logit_weight)
        self.use_event_responsibility_head = bool(use_event_responsibility_head)
        self.event_responsibility_hidden = int(event_responsibility_hidden)
        self.event_responsibility_detach_features = bool(event_responsibility_detach_features)
        self.use_event_route_head = bool(use_event_route_head)
        self.event_route_hidden = int(event_route_hidden)
        self.event_route_init = float(event_route_init)
        self.event_route_detach_inputs = bool(event_route_detach_inputs)
        self.event_route_response_penalty = max(0.0, float(event_route_response_penalty))
        self.use_response_suppressor_head = bool(use_response_suppressor_head)
        self.response_suppressor_hidden = int(response_suppressor_hidden)
        self.response_suppressor_detach_inputs = bool(response_suppressor_detach_inputs)
        self.use_evidence_fusion_head = bool(use_evidence_fusion_head)
        self.evidence_fusion_hidden = int(evidence_fusion_hidden)
        self.evidence_fusion_detach_inputs = bool(evidence_fusion_detach_inputs)
        self.evidence_fusion_use_graph_response = bool(
            evidence_fusion_use_graph_response
        )
        self.use_source_interaction_head = bool(use_source_interaction_head)
        self.source_interaction_detach_inputs = bool(source_interaction_detach_inputs)
        self.use_source_consistency_head = bool(use_source_consistency_head)
        self.source_consistency_detach_inputs = bool(source_consistency_detach_inputs)
        self.use_channel_temporal_corefinement = bool(use_channel_temporal_corefinement)
        self.corefinement_init = float(corefinement_init)
        self.corefinement_detach_first_pass = bool(corefinement_detach_first_pass)
        self.use_source_aware_corefinement = bool(use_source_aware_corefinement)
        self.source_aware_corefinement_init = float(source_aware_corefinement_init)
        self.source_aware_corefinement_detach_gate = bool(source_aware_corefinement_detach_gate)
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
        self.register_buffer(
            'channel_mechanism_score_center',
            torch.zeros(1),
            persistent=False,
        )
        self.register_buffer(
            'channel_mechanism_score_scale',
            torch.ones(1),
            persistent=False,
        )
        self.register_buffer(
            'channel_graph_reliability',
            torch.ones(enc_in),
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

        if self.channel_graph_type == "directed_sparse":
            self.channel_graph = DirectedSparseChannelGraph(
                num_nodes=enc_in,
                embedding_dim=self.directed_graph_embedding_dim,
                parent_topk=self.directed_graph_parent_topk,
                dropout=dropout,
                use_static_prior=self.use_channel_corr_prior,
                static_prior_weight=self.channel_corr_prior_weight,
                static_prior_bias=self.channel_corr_prior_bias,
                use_signed_transfer=self.directed_graph_use_signed_transfer,
                transfer_mode=self.directed_graph_transfer_mode,
                transfer_rank=self.directed_graph_transfer_rank,
                transfer_scale=self.directed_graph_transfer_scale,
            )
        elif self.channel_graph_type == "adaptive_symmetric":
            self.channel_graph = ChannelAdaptiveGraph(
                num_nodes=enc_in, topk=topk,
                sparse_topk=sparse_topk, dropout=dropout,
                use_static_prior=self.use_channel_corr_prior,
                static_prior_weight=self.channel_corr_prior_weight,
                static_prior_bias=self.channel_corr_prior_bias,
            )
        else:
            raise ValueError(f"Unsupported channel_graph_type={self.channel_graph_type!r}")

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

        coupling_init = min(max(float(mechanism_coupling_init), 1e-3), 1.0 - 1e-3)
        self.mechanism_coupling_logit = nn.Parameter(
            torch.tensor(float(np.log(coupling_init / (1.0 - coupling_init))))
        )
        source_preserve_init = min(max(float(source_preserving_init), 1e-3), 1.0 - 1e-3)
        self.source_preserving_logit = nn.Parameter(
            torch.tensor(float(np.log(source_preserve_init / (1.0 - source_preserve_init))))
        )
        self.mechanism_context_fusion = nn.Sequential(
            nn.Linear(c_out * 3, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )
        feedback_init = min(max(float(mechanism_feedback_init), 1e-3), 1.0 - 1e-3)
        self.mechanism_feedback_logit = nn.Parameter(
            torch.tensor(float(np.log(feedback_init / (1.0 - feedback_init))))
        )

        predictive_blend_init = min(max(float(mechanism_predictive_blend_init), 1e-3), 1.0 - 1e-3)
        self.mechanism_predictive_blend_logit = nn.Parameter(
            torch.tensor(float(np.log(predictive_blend_init / (1.0 - predictive_blend_init))))
        )
        self.mechanism_predictor = nn.Sequential(
            nn.Linear(c_out * 3, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )
        self.mechanism_predictive_fusion = nn.Sequential(
            nn.Linear(c_out * 3, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )

        corefine_init = min(max(float(corefinement_init), 1e-3), 1.0 - 1e-3)
        self.corefinement_logit = nn.Parameter(
            torch.tensor(float(np.log(corefine_init / (1.0 - corefine_init))))
        )
        self.corefinement_fusion = nn.Sequential(
            nn.Linear(c_out * 4, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )

        # === ★ P0: Dual-Path VQ Bottleneck (旁路模式) ===
        source_corefine_init = min(
            max(float(source_aware_corefinement_init), 1e-3),
            1.0 - 1e-3,
        )
        self.source_aware_corefinement_logit = nn.Parameter(
            torch.tensor(float(np.log(source_corefine_init / (1.0 - source_corefine_init))))
        )
        self.source_aware_corefinement_fusion = nn.Sequential(
            nn.Linear(c_out * 5, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )

        source_gate_init = min(max(float(source_gate_init), 1e-3), 1.0 - 1e-3)
        source_gate_bias = float(np.log(source_gate_init / (1.0 - source_gate_init)))
        source_gate_feature_count = 3 + int(self.use_onset_aware_source_gate)
        # The delayed cross-channel explanation gain is a response cue. It is
        # learned inside the source gate rather than applied as a hand-tuned
        # export penalty.
        source_gate_feature_count += int(self.use_causal_response_evidence)
        source_gate_input_dim = c_out * source_gate_feature_count
        self.source_gate_net = nn.Sequential(
            nn.Linear(source_gate_input_dim, c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_out, c_out),
        )
        nn.init.zeros_(self.source_gate_net[-1].weight)
        nn.init.zeros_(self.source_gate_net[-1].bias)
        self.source_gate_bias = nn.Parameter(torch.full((1, 1, c_out), source_gate_bias))

        if self.use_dual_expert_fusion:
            self.dual_expert_gate = DualExpertFusionGate(
                feature_dim=5,
                hidden=self.dual_expert_hidden,
                dropout=dropout,
                init_graph_weight=self.dual_expert_graph_init,
            )
        else:
            self.dual_expert_gate = None

        if self.use_source_expert_branch:
            if self.source_expert_gate_scope == "event":
                self.source_expert_gate = EventExpertGate(
                    feature_dim=5,
                    hidden=self.source_expert_hidden,
                    dropout=dropout,
                    init_graph_weight=self.source_expert_graph_init,
                )
            else:
                self.source_expert_gate = DualExpertFusionGate(
                    feature_dim=5,
                    hidden=self.source_expert_hidden,
                    dropout=dropout,
                    init_graph_weight=self.source_expert_graph_init,
                )
            self.source_expert_head = SourceExpertHead(
                feature_dim=5,
                hidden=self.source_expert_hidden,
                dropout=dropout,
            )
        else:
            self.source_expert_gate = None
            self.source_expert_head = None

        root_hidden = max(16, min(64, c_out))
        if self.root_score_head_mode in {
            "root_response",
            "root-response",
            "response_decomposition",
        }:
            head_kwargs = dict(
                source_dim=(9 if self.root_response_use_innovation_split else 8)
                + int(self.use_source_expert_branch),
                response_dim=(2 if self.root_response_use_innovation_split else 3)
                + int(self.use_causal_response_evidence),
                hidden=root_hidden,
                dropout=dropout,
                response_penalty_init=self.root_response_penalty_init,
                source_confidence_discount=self.root_response_confidence_discount,
            )
            if self.use_pairwise_root_response_head:
                self.root_score_head = PairwiseRootResponseRCAHead(
                    **head_kwargs,
                    channel=c_out,
                    relation_rank=self.pairwise_root_response_rank,
                    graph_weight_init=self.pairwise_root_response_graph_weight,
                    explanation_reward_init=self.pairwise_root_response_reward_init,
                    response_pair_penalty_init=self.pairwise_root_response_penalty_init,
                    pairwise_logit_weight=self.pairwise_root_response_logit_weight,
                )
            else:
                self.root_score_head = RootResponseRCAHead(**head_kwargs)
        else:
            self.root_score_head = nn.Sequential(
                nn.Linear(7, root_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(root_hidden, 1),
            )
            nn.init.constant_(self.root_score_head[-1].bias, -2.0)

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

        if self.use_strict_cross_mechanism:
            self.strict_cross_mechanism = StrictCrossLagMechanism(
                channel=c_out,
                lags=self.strict_cross_lags,
                topk=self.strict_cross_topk,
                dropout=dropout,
                use_static_prior=self.strict_cross_use_channel_prior,
            )
        else:
            self.strict_cross_mechanism = None

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

        if self.use_synthetic_rca_head:
            hidden = max(64, d_model)
            self.synthetic_rca_head = nn.Sequential(
                nn.Linear(d_model * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, c_out),
            )
        else:
            self.synthetic_rca_head = None

        if self.use_event_responsibility_head:
            self.event_responsibility_head = EventResponsibilityHead(
                feature_dim=8,
                hidden=self.event_responsibility_hidden,
                dropout=dropout,
            )
        else:
            self.event_responsibility_head = None

        if self.use_sps_role_head:
            self.sps_role_head = SPSRoleHead(
                feature_dim=4,
                hidden=self.sps_role_hidden,
                dropout=dropout,
            )
        else:
            self.sps_role_head = None

        if self.use_event_route_head:
            self.event_route_head = EventRouteHead(
                feature_dim=9,
                hidden=self.event_route_hidden,
                dropout=dropout,
                init_alpha=self.event_route_init,
            )
        else:
            self.event_route_head = None

        if self.use_response_suppressor_head:
            self.response_suppressor_head = ResponseSuppressorHead(
                feature_dim=7,
                hidden=self.response_suppressor_hidden,
                dropout=dropout,
            )
        else:
            self.response_suppressor_head = None

        if self.use_evidence_fusion_head:
            with torch.random.fork_rng(devices=[]):
                self.evidence_fusion_head = EvidenceFusionHead(
                    evidence_dim=10
                    if self.evidence_fusion_use_graph_response
                    else 8,
                    hidden=self.evidence_fusion_hidden,
                    dropout=0.0,
                )
        else:
            self.evidence_fusion_head = None

        if self.use_source_interaction_head:
            self.source_interaction_head = SourceInteractionHead(evidence_dim=5)
        else:
            self.source_interaction_head = None

        if self.use_source_consistency_head:
            self.source_consistency_head = SourceConsistencyHead(evidence_dim=6)
        else:
            self.source_consistency_head = None

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
        self._set_trainable(self.strict_cross_mechanism, self.use_strict_cross_mechanism)
        self._set_trainable(self.synthetic_anomaly_head, self.use_synthetic_anomaly_head)
        self._set_trainable(self.synthetic_rca_head, self.use_synthetic_rca_head)
        self._set_trainable(self.vq_bottleneck, self.use_vq_bypass)
        self._set_trainable(self.multi_scale_scorer, self.use_multi_scale_scorer)
        self._set_trainable(self.mechanism_context_fusion, self.use_mechanism_coupled_decoder)
        self.mechanism_coupling_logit.requires_grad = self.use_mechanism_coupled_decoder
        self.source_preserving_logit.requires_grad = (
            self.use_mechanism_coupled_decoder and self.use_source_preserving_decoder
        )
        self.mechanism_feedback_logit.requires_grad = self.use_mechanism_residual_feedback
        self._set_trainable(self.mechanism_predictor, self.use_mechanism_predictive_head)
        self._set_trainable(self.mechanism_predictive_fusion, self.use_mechanism_predictive_head)
        self.mechanism_predictive_blend_logit.requires_grad = self.use_mechanism_predictive_head
        self._set_trainable(self.source_gate_net, self.use_source_gate)
        self.source_gate_bias.requires_grad = self.use_source_gate
        self._set_trainable(self.dual_expert_gate, self.use_dual_expert_fusion)
        self._set_trainable(self.source_expert_gate, self.use_source_expert_branch)
        self._set_trainable(self.source_expert_head, self.use_source_expert_branch)
        self._set_trainable(self.root_score_head, self.use_root_score_head)
        self._set_trainable(
            self.event_responsibility_head,
            self.use_event_responsibility_head,
        )
        self._set_trainable(self.sps_role_head, self.use_sps_role_head)
        self._set_trainable(self.event_route_head, self.use_event_route_head)
        self._set_trainable(
            self.response_suppressor_head,
            self.use_response_suppressor_head,
        )
        self._set_trainable(self.evidence_fusion_head, self.use_evidence_fusion_head)
        self._set_trainable(
            self.source_interaction_head,
            self.use_source_interaction_head,
        )
        self._set_trainable(
            self.source_consistency_head,
            self.use_source_consistency_head,
        )
        self._set_trainable(self.corefinement_fusion, self.use_channel_temporal_corefinement)
        self.corefinement_logit.requires_grad = self.use_channel_temporal_corefinement
        self._set_trainable(
            self.source_aware_corefinement_fusion,
            self.use_source_aware_corefinement,
        )
        self.source_aware_corefinement_logit.requires_grad = (
            self.use_source_aware_corefinement
        )

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

    def _source_onset_feature(self, resid):
        """Causal local innovation used as an optional source-gate cue."""
        abs_resid = resid.abs()
        if abs_resid.shape[1] <= 1:
            return torch.zeros_like(abs_resid)
        shifted = torch.cat(
            [torch.zeros_like(abs_resid[:, :1, :]), abs_resid[:, :-1, :]],
            dim=1,
        )
        window = max(1, int(self.source_gate_onset_window))
        if window > 1:
            shifted_t = shifted.transpose(1, 2)
            shifted_t = F.pad(shifted_t, (window - 1, 0), mode="replicate")
            baseline = F.avg_pool1d(
                shifted_t,
                kernel_size=window,
                stride=1,
            ).transpose(1, 2)
        else:
            baseline = shifted
        delta = (abs_resid - baseline).clamp_min(0.0)
        scale = (
            baseline.detach().abs()
            + abs_resid.detach().mean(dim=1, keepdim=True)
            + 1e-6
        )
        return torch.tanh(delta / scale)

    def get_sparse_loss(self):
        """通道图 L1 稀疏正则化损失"""
        if not self.use_channel_graph:
            return next(self.parameters()).new_tensor(0.0)
        return self.channel_graph.get_l1_penalty()

    def set_warmup_progress(self, alpha: float):
        if not self.use_channel_graph:
            return
        self.channel_graph.set_warmup_progress(alpha)

    def set_channel_static_prior(self, prior):
        if not self.use_channel_graph or not hasattr(self.channel_graph, "set_static_prior"):
            return
        self.channel_graph.set_static_prior(prior)

    def set_strict_cross_static_prior(self, prior):
        if self.strict_cross_mechanism is None:
            return
        self.strict_cross_mechanism.set_static_prior(prior)

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

    def set_channel_mechanism_score_stats(self, score_center, score_scale):
        self.channel_mechanism_score_center.copy_(
            torch.as_tensor(
                [score_center],
                dtype=self.channel_mechanism_score_center.dtype,
                device=self.channel_mechanism_score_center.device,
            )
        )
        self.channel_mechanism_score_scale.copy_(
            torch.as_tensor(
                [score_scale],
                dtype=self.channel_mechanism_score_scale.dtype,
                device=self.channel_mechanism_score_scale.device,
            ).clamp_min(self.channel_mechanism_score_eps)
        )

    def _normalize_channel_mechanism_score(self, mechanism_score):
        center = self.channel_mechanism_score_center.to(
            device=mechanism_score.device, dtype=mechanism_score.dtype,
        )
        scale = self.channel_mechanism_score_scale.to(
            device=mechanism_score.device, dtype=mechanism_score.dtype,
        )
        return (mechanism_score - center).clamp_min(0.0) / scale.clamp_min(self.channel_mechanism_score_eps)

    def set_channel_graph_reliability(self, reliability):
        reliability = torch.as_tensor(
            reliability,
            dtype=self.channel_graph_reliability.dtype,
            device=self.channel_graph_reliability.device,
        ).flatten()
        if reliability.numel() != self.channel_graph_reliability.numel():
            raise ValueError(
                "channel graph reliability size mismatch: "
                f"expected {self.channel_graph_reliability.numel()}, got {reliability.numel()}"
            )
        self.channel_graph_reliability.copy_(reliability.clamp(0.0, 1.0))

    def _graph_neighbor_context(self, feat, A_adaptive):
        _, _, C = feat.shape
        eye = torch.eye(C, device=feat.device, dtype=feat.dtype).unsqueeze(0)
        A_mech = A_adaptive.to(dtype=feat.dtype) * (1.0 - eye)
        col_sum = A_mech.sum(dim=1, keepdim=True).clamp_min(1e-8)
        A_mech = A_mech / col_sum
        if (
            self.channel_graph_type == "directed_sparse"
            and self.directed_graph_use_signed_transfer
            and hasattr(self.channel_graph, "predict_context")
        ):
            parent_context = self.channel_graph.predict_context(feat, A_mech)
        else:
            parent_context = torch.bmm(feat, A_mech)
        # This context is used to predict a target channel from *other* channels.
        # Reliability must not interpolate it with ``feat``: a zero-reliability
        # channel would otherwise receive its own target as an input, defeating
        # the self-exclusion above and making train-time and calibrated inference
        # follow different information paths. Reliability is applied only when
        # graph evidence is propagated to explain a response.
        return parent_context

    def _graph_propagation_context(
        self,
        evidence,
        A_adaptive,
        target_evidence=None,
    ):
        _, _, C = evidence.shape
        eye = torch.eye(C, device=evidence.device, dtype=evidence.dtype).unsqueeze(0)
        adjacency = A_adaptive.to(dtype=evidence.dtype) * (1.0 - eye)
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)
        def propagate_once(lagged_evidence):
            if (
                self.channel_graph_type == "directed_sparse"
                and hasattr(self.channel_graph, "propagate_evidence")
            ):
                return self.channel_graph.propagate_evidence(
                    lagged_evidence, adjacency
                )
            return torch.bmm(lagged_evidence, adjacency)

        if self.use_multilag_graph_propagation:
            lagged_responses = []
            for lag in self.graph_propagation_lags:
                if lag <= 0:
                    shifted = evidence
                elif lag >= evidence.shape[1]:
                    shifted = torch.zeros_like(evidence)
                else:
                    shifted = torch.zeros_like(evidence)
                    shifted[:, lag:, :] = evidence[:, :-lag, :]
                lagged_responses.append(propagate_once(shifted))
            response_stack = torch.stack(lagged_responses, dim=-1)
            if (
                self.graph_propagation_lag_mode == "alignment"
                and target_evidence is not None
            ):
                target = target_evidence.to(dtype=evidence.dtype)
                target_centered = target - target.mean(dim=1, keepdim=True)
                response_centered = response_stack - response_stack.mean(
                    dim=1, keepdim=True
                )
                numerator = (
                    response_centered * target_centered.unsqueeze(-1)
                ).sum(dim=1)
                energy_product = (
                    response_centered.pow(2).sum(dim=1)
                    * target_centered.pow(2).sum(dim=1).unsqueeze(-1)
                )
                denominator = energy_product.clamp_min(1e-8).sqrt()
                alignment = torch.nan_to_num(
                    numerator / denominator,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                temperature = F.softplus(
                    self.graph_propagation_alignment_temp_raw
                ).clamp_min(0.05)
                lag_weights = torch.softmax(
                    alignment / temperature,
                    dim=-1,
                )
                propagated = (
                    response_stack * lag_weights.unsqueeze(1)
                ).sum(dim=-1)
            else:
                if self.graph_propagation_lag_logits is None:
                    lag_weights = evidence.new_full(
                        (C, len(self.graph_propagation_lags)),
                        1.0 / len(self.graph_propagation_lags),
                    )
                else:
                    lag_weights = torch.softmax(
                        self.graph_propagation_lag_logits,
                        dim=-1,
                    ).to(device=evidence.device, dtype=evidence.dtype)
                propagated = (
                    response_stack * lag_weights.view(1, 1, C, -1)
                ).sum(dim=-1)
            self._last_graph_propagation_lag_weights = lag_weights
        else:
            propagated = propagate_once(evidence)
            self._last_graph_propagation_lag_weights = None
        if not self.use_channel_graph_reliability:
            return propagated
        reliability = self.channel_graph_reliability.to(
            device=evidence.device,
            dtype=evidence.dtype,
        ).view(1, 1, C)
        return reliability * propagated

    def _source_branch_mechanism_error(
        self,
        resid,
        A_adaptive,
        channel_mechanism_error,
    ):
        if self.channel_graph_role == "response_only":
            return torch.zeros_like(resid)
        if channel_mechanism_error is not None:
            return channel_mechanism_error.to(dtype=resid.dtype)
        if self.use_channel_graph:
            return torch.abs(resid - self._graph_neighbor_context(resid, A_adaptive))
        return torch.zeros_like(resid)

    def _channel_mechanism(self, resid, A_adaptive):
        _, _, C = resid.shape
        pred = self._graph_neighbor_context(resid, A_adaptive)
        err = torch.abs(resid - pred)
        k = min(max(1, self.score_topk_k or self.multi_scale_scorer.topk_k), C)
        score = err.topk(k=k, dim=-1, largest=True, sorted=False)[0].mean(dim=-1)
        loss = F.smooth_l1_loss(pred, resid)
        return pred, score, loss, err

    def _mechanism_predictive(self, resid, A_adaptive, temporal_pred=None):
        parent_context = self._graph_neighbor_context(resid, A_adaptive)
        if temporal_pred is None:
            temporal_context = torch.zeros_like(resid)
        else:
            temporal_context = temporal_pred.to(dtype=resid.dtype)
            if temporal_context.shape[1] != resid.shape[1]:
                if temporal_context.shape[1] > resid.shape[1]:
                    temporal_context = temporal_context[:, -resid.shape[1]:, :]
                else:
                    pad_len = resid.shape[1] - temporal_context.shape[1]
                    temporal_context = F.pad(
                        temporal_context.transpose(1, 2),
                        (pad_len, 0),
                        mode="constant",
                        value=0.0,
                    ).transpose(1, 2)

        predictor_input = torch.cat(
            [
                parent_context,
                temporal_context,
                (parent_context - temporal_context).abs(),
            ],
            dim=-1,
        )
        pred = self.mechanism_predictor(predictor_input)
        err = torch.abs(resid - pred)
        k = min(max(1, self.score_topk_k or self.multi_scale_scorer.topk_k), resid.shape[-1])
        score = err.topk(k=k, dim=-1, largest=True, sorted=False)[0].mean(dim=-1)
        loss = F.smooth_l1_loss(pred, resid)
        return pred, score, loss, err

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

    def forward(self, x, return_root_score=False):
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
        causal_pred = None
        causal_channel_error = None
        causal_response_support = None
        strict_cross_pred = None
        strict_cross_self_pred = None
        strict_cross_attention = None
        strict_cross_edge_weight = None
        strict_cross_full_error = None
        strict_cross_self_error = None
        strict_cross_response_gain = None
        strict_cross_mechanism_loss = None
        sps_root_logits = None
        sps_response_logits = None
        channel_mechanism_score = None
        channel_mechanism_loss = None
        channel_mechanism_error = None
        channel_mechanism_pred = None
        mechanism_coupling_weight = None
        source_preserving_weight = None
        mechanism_predictive_blend_weight = None
        corefinement_weight = None
        source_aware_corefinement_weight = None
        source_gate = None
        source_gate_score = None
        source_gate_onset_feature = None
        dual_expert_gate = None
        dual_expert_independent_evidence = None
        dual_expert_graph_evidence = None
        dual_expert_independent_feat = None
        dual_expert_graph_feat = None
        source_expert_graph_gate = None
        source_expert_logits = None
        source_expert_independent_feat = None
        source_expert_graph_feat = None
        root_score_logits = None
        root_response_source_evidence = None
        root_response_response_evidence = None
        root_response_source_confidence = None
        root_response_effective_response_evidence = None
        root_response_propagation_support = None
        root_response_pairwise_relation = None
        root_response_pairwise_source_support = None
        root_response_pairwise_response_support = None
        event_responsibility_logits = None
        event_route_logits = None
        event_route_alpha = None
        event_route_score = None
        event_route_graph_score = None
        event_route_event_score = None
        event_route_response_score = None
        response_suppressor_logits = None
        evidence_fusion_logits = None
        evidence_fusion_weights = None
        source_interaction_logits = None
        source_interaction_weights = None
        source_consistency_logits = None
        source_consistency_weights = None
        source_consistency_support = None

        if self.use_lagged_causal_graph and self.lagged_causal_graph is not None:
            causal_input = resid.detach() if self.causal_detach_backbone else resid
            (
                causal_pred,
                causal_score,
                _,
                causal_self_pred,
            ) = self.lagged_causal_graph(causal_input)
            valid_start = self.lagged_causal_graph.max_lag
            if L > valid_start:
                causal_channel_error = F.smooth_l1_loss(
                    causal_pred,
                    causal_input,
                    reduction="none",
                )
                causal_self_error = F.smooth_l1_loss(
                    causal_self_pred,
                    causal_input,
                    reduction="none",
                )
                # A response is only credited when delayed cross-channel
                # parents improve over the channel's own temporal history.
                # Both predictors are causal and never access x_i(t).
                causal_response_support = (
                    causal_self_error - causal_channel_error
                ).clamp_min(0.0)
                causal_mechanism_loss = F.smooth_l1_loss(
                    causal_pred[:, valid_start:, :],
                    causal_input[:, valid_start:, :],
                )

        if self.use_strict_cross_mechanism and self.strict_cross_mechanism is not None:
            strict_input = resid.detach() if self.strict_cross_detach_backbone else resid
            (
                strict_cross_pred,
                strict_cross_self_pred,
                strict_cross_attention,
                strict_cross_edge_weight,
            ) = self.strict_cross_mechanism(strict_input)
            strict_start = self.strict_cross_mechanism.max_lag
            strict_cross_full_error = F.smooth_l1_loss(
                strict_cross_pred,
                strict_input,
                reduction="none",
            )
            strict_cross_self_error = F.smooth_l1_loss(
                strict_cross_self_pred,
                strict_input,
                reduction="none",
            )
            strict_cross_response_gain = (
                strict_cross_self_error - strict_cross_full_error
            ).clamp_min(0.0)
            if L > strict_start:
                strict_cross_mechanism_loss = F.smooth_l1_loss(
                    strict_cross_pred[:, strict_start:, :],
                    strict_input[:, strict_start:, :],
                )

        # 步骤 2: 自适应通道依赖图
        if self.use_channel_graph:
            graph_adapted, A_adaptive = self.channel_graph(resid)
            resid_adapted = (
                resid if self.channel_graph_evidence_only else graph_adapted
            )
        else:
            resid_adapted = resid
            A_adaptive = torch.eye(C, device=x.device).unsqueeze(0).expand(B, C, C)

        if self.use_mechanism_predictive_head and self.use_channel_graph:
            (
                channel_mechanism_pred,
                channel_mechanism_score,
                channel_mechanism_loss,
                channel_mechanism_error,
            ) = self._mechanism_predictive(resid, A_adaptive, temporal_pred=causal_pred)
        elif self.use_channel_graph and (
            self.use_channel_mechanism_score or self.lambda_channel_mechanism > 0
        ):
            (
                channel_mechanism_pred,
                channel_mechanism_score,
                channel_mechanism_loss,
                channel_mechanism_error,
            ) = self._channel_mechanism(resid, A_adaptive)

        # 步骤 3: 简化时序图
        if self.use_source_gate and self.use_channel_graph:
            neighbor_context = (
                channel_mechanism_pred
                if channel_mechanism_pred is not None
                else self._graph_neighbor_context(resid, A_adaptive)
            )
            if self.channel_graph_role == "response_only":
                channel_deviation = torch.abs(resid)
            else:
                channel_deviation = (
                    channel_mechanism_error
                    if channel_mechanism_error is not None
                    else torch.abs(resid - neighbor_context)
                )
            causal_deviation = (
                causal_channel_error
                if causal_channel_error is not None
                else torch.zeros_like(channel_deviation)
            ).to(dtype=resid.dtype)
            gate_parts = [
                resid,
                channel_deviation.to(dtype=resid.dtype),
                causal_deviation,
            ]
            if self.use_causal_response_evidence:
                gate_parts.append(
                    (
                        causal_response_support
                        if causal_response_support is not None
                        else torch.zeros_like(channel_deviation)
                    ).to(dtype=resid.dtype)
                )
            if self.use_onset_aware_source_gate:
                source_gate_onset_feature = self._source_onset_feature(resid).to(
                    dtype=resid.dtype
                )
                gate_parts.append(source_gate_onset_feature)
            gate_input = torch.cat(gate_parts, dim=-1)
            source_gate = torch.sigmoid(self.source_gate_net(gate_input) + self.source_gate_bias)
            source_gate_evidence = channel_deviation.to(dtype=resid.dtype) + causal_deviation
            if source_gate_onset_feature is not None and self.source_gate_onset_weight > 0.0:
                source_gate_evidence = (
                    source_gate_evidence
                    + self.source_gate_onset_weight * source_gate_onset_feature
                )
            source_gate_score = source_gate * source_gate_evidence
            if not self.channel_graph_evidence_only:
                resid_adapted = source_gate * resid + (1.0 - source_gate) * resid_adapted

        if (
            self.use_channel_temporal_corefinement
            and self.use_channel_graph
            and self.use_temporal_graph
            and not self.channel_graph_evidence_only
        ):
            if self.corefinement_detach_first_pass:
                with torch.no_grad():
                    resid_serial, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
                resid_serial = resid_serial.detach()
            else:
                resid_serial, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
            temporal_mechanism_context = self._graph_neighbor_context(resid_serial, A_adaptive)
            if self.use_source_aware_corefinement and source_gate is not None:
                source_refine_gate = (
                    source_gate.detach()
                    if self.source_aware_corefinement_detach_gate
                    else source_gate
                )
                source_context = self._graph_neighbor_context(
                    source_refine_gate * resid_serial,
                    A_adaptive,
                )
                corefine_delta = self.source_aware_corefinement_fusion(
                    torch.cat(
                        [
                            resid_serial,
                            resid_adapted,
                            temporal_mechanism_context,
                            source_context,
                            (resid_serial - source_context).abs(),
                        ],
                        dim=-1,
                    )
                )
                source_aware_corefinement_weight = torch.sigmoid(
                    self.source_aware_corefinement_logit
                )
                resid_corefined = resid_serial + source_aware_corefinement_weight * corefine_delta
            else:
                corefine_delta = self.corefinement_fusion(
                    torch.cat(
                        [
                            resid_serial,
                            resid_adapted,
                            temporal_mechanism_context,
                            (resid_serial - temporal_mechanism_context).abs(),
                        ],
                        dim=-1,
                    )
                )
                corefinement_weight = torch.sigmoid(self.corefinement_logit)
                resid_corefined = resid_serial + corefinement_weight * corefine_delta
            stage1_feat, A_temp = self.temporal_graph(resid_corefined, A_proximity=A_adaptive)
        elif (
            self.use_state_aware_fusion
            and self.use_channel_graph
            and self.use_temporal_graph
            and not self.channel_graph_evidence_only
        ):
            resid_serial, A_temp = self.temporal_graph(resid_adapted, A_proximity=A_adaptive)
            resid_temporal_base, _ = self.temporal_graph(resid, A_proximity=A_adaptive)
            stage1_feat, state_aware_info = self.state_aware_fusion(
                resid,
                resid_adapted,
                resid_temporal_base,
                resid_serial=resid_serial,
            )
        elif (
            self.use_parallel_graph_fusion
            and self.use_channel_graph
            and self.use_temporal_graph
            and not self.channel_graph_evidence_only
        ):
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
                backbone_graph = None if self.channel_graph_evidence_only else A_adaptive
                resid_temp, A_temp = self.temporal_graph(
                    resid_adapted,
                    A_proximity=backbone_graph,
                )
            else:
                resid_temp = resid_adapted
                A_temp = None
            stage1_feat = resid_temp  # (B, L, C)

        # Keep a graph-independent temporal expert alongside the graph-conditioned
        # path. Their mixture is produced before the shared encoder/decoder, so the
        # two explanations compete in representation space rather than at RCA export.
        if (
            self.use_dual_expert_fusion
            and self.dual_expert_gate is not None
            and self.use_channel_graph
            and self.use_temporal_graph
            and not self.channel_graph_evidence_only
        ):
            dual_expert_graph_feat = stage1_feat
            dual_expert_independent_feat, _ = self.temporal_graph(
                resid,
                A_proximity=None,
            )
            dual_expert_independent_evidence = torch.abs(
                resid - dual_expert_independent_feat
            )
            dual_expert_graph_evidence = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.abs(resid - dual_expert_graph_feat)
            )
            gate_source_feature = (
                source_gate.to(dtype=resid.dtype)
                if source_gate is not None
                else torch.zeros_like(resid)
            )
            dual_gate_features = torch.stack(
                [
                    resid.abs().mean(dim=-1),
                    dual_expert_independent_evidence.mean(dim=-1),
                    dual_expert_graph_evidence.abs().mean(dim=-1),
                    torch.abs(
                        dual_expert_graph_feat - dual_expert_independent_feat
                    ).mean(dim=-1),
                    gate_source_feature.mean(dim=-1),
                ],
                dim=-1,
            )
            if self.training and self.dual_expert_detach_inputs:
                dual_gate_features = dual_gate_features.detach()
            dual_expert_gate = self.dual_expert_gate(dual_gate_features)
            stage1_feat = (
                dual_expert_gate * dual_expert_graph_feat
                + (1.0 - dual_expert_gate) * dual_expert_independent_feat
            )

        # Source attribution uses a separate graph-conditioned expert while the
        # reconstruction path above remains unchanged. This prevents graph
        # propagation from shifting anomaly-event boundaries.
        if (
            self.use_source_expert_branch
            and self.source_expert_gate is not None
            and self.source_expert_head is not None
            and self.use_channel_graph
            and self.use_temporal_graph
        ):
            if self.channel_graph_evidence_only:
                source_expert_independent_feat = stage1_feat
            else:
                source_expert_independent_feat, _ = self.temporal_graph(
                    resid,
                    A_proximity=None,
                )
            if self.training and self.source_expert_gradient_checkpoint:
                source_expert_graph_feat = checkpoint(
                    lambda graph_input, adjacency: self.temporal_graph(
                        graph_input,
                        A_proximity=adjacency,
                    )[0],
                    graph_adapted,
                    A_adaptive,
                    use_reentrant=False,
                )
            else:
                source_expert_graph_feat, _ = self.temporal_graph(
                    graph_adapted,
                    A_proximity=A_adaptive,
                )
            source_expert_graph_evidence = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.abs(resid - source_expert_graph_feat)
            )
            source_expert_gate_feature = (
                source_gate.to(dtype=resid.dtype)
                if source_gate is not None
                else torch.zeros_like(resid)
            )
            source_expert_propagation = self._graph_neighbor_context(
                source_expert_graph_evidence,
                A_adaptive,
            ).abs()
            source_expert_gate_features = torch.stack(
                [
                    source_expert_graph_evidence.abs().mean(dim=-1),
                    torch.abs(resid - source_expert_independent_feat).mean(dim=-1),
                    source_expert_propagation.mean(dim=-1),
                    torch.abs(
                        source_expert_graph_feat - source_expert_independent_feat
                    ).mean(dim=-1),
                    source_expert_gate_feature.mean(dim=-1),
                ],
                dim=-1,
            )
            if self.training and self.source_expert_detach_gate_inputs:
                source_expert_gate_features = source_expert_gate_features.detach()
            source_expert_graph_gate = self.source_expert_gate(
                source_expert_gate_features
            )
            source_expert_fused_feat = (
                source_expert_graph_gate * source_expert_graph_feat
                + (1.0 - source_expert_graph_gate)
                * source_expert_independent_feat
            )
            source_expert_features = torch.stack(
                [
                    source_expert_fused_feat,
                    source_expert_independent_feat,
                    source_expert_graph_feat,
                    torch.abs(source_expert_graph_feat - source_expert_independent_feat),
                    source_expert_graph_evidence,
                ],
                dim=-1,
            )
            source_expert_logits = self.source_expert_head(source_expert_features)

        if (
            self.use_mechanism_coupled_decoder
            and self.use_channel_graph
            and not self.channel_graph_evidence_only
        ):
            mechanism_context = self._graph_neighbor_context(stage1_feat, A_adaptive)
            mechanism_delta = self.mechanism_context_fusion(
                torch.cat(
                    [
                        stage1_feat,
                        mechanism_context,
                        (stage1_feat - mechanism_context).abs(),
                    ],
                    dim=-1,
                )
            )
            mechanism_coupling_weight = torch.sigmoid(self.mechanism_coupling_logit)
            if self.use_source_preserving_decoder and source_gate is not None:
                source_preserve_gate = (
                    source_gate.detach()
                    if self.source_preserving_detach_gate
                    else source_gate
                )
                source_preserving_weight = torch.sigmoid(self.source_preserving_logit)
                local_coupling = mechanism_coupling_weight * (
                    1.0 - source_preserving_weight * source_preserve_gate
                )
                if dual_expert_gate is not None:
                    local_coupling = dual_expert_gate * local_coupling
                stage1_feat = stage1_feat + local_coupling * mechanism_delta
            else:
                coupling = mechanism_coupling_weight
                if dual_expert_gate is not None:
                    coupling = dual_expert_gate * coupling
                stage1_feat = stage1_feat + coupling * mechanism_delta

        mechanism_feedback_weight = None
        if (
            self.use_mechanism_residual_feedback
            and self.use_channel_graph
            and channel_mechanism_pred is not None
            and not self.channel_graph_evidence_only
        ):
            mechanism_residual = resid - channel_mechanism_pred.to(dtype=resid.dtype)
            feedback = mechanism_residual.detach() if self.mechanism_feedback_detach else mechanism_residual
            feedback_direction = feedback
            if self.mechanism_feedback_norm == "sample_z":
                center = feedback.mean(dim=(1, 2), keepdim=True)
                scale = feedback.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
                normalized_feedback = (feedback - center) / scale
            elif self.mechanism_feedback_norm == "none":
                normalized_feedback = feedback
            else:
                scale = feedback.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                normalized_feedback = feedback / scale
            if self.mechanism_feedback_norm == "none":
                confidence = torch.ones_like(feedback_direction)
            else:
                confidence = torch.sigmoid(normalized_feedback.abs() - 1.0)
            clip = max(float(self.mechanism_feedback_clip), 0.0)
            if clip > 0:
                feedback_direction = feedback_direction.clamp(min=-clip, max=clip)
            feedback = confidence * feedback_direction
            mechanism_feedback_weight = torch.sigmoid(self.mechanism_feedback_logit)
            feedback_weight = mechanism_feedback_weight
            if dual_expert_gate is not None:
                feedback_weight = dual_expert_gate * feedback_weight
            stage1_feat = stage1_feat + feedback_weight * feedback

        if (
            self.use_mechanism_predictive_head
            and channel_mechanism_pred is not None
            and not self.channel_graph_evidence_only
        ):
            mechanism_delta = self.mechanism_predictive_fusion(
                torch.cat(
                    [
                        stage1_feat,
                        channel_mechanism_pred,
                        (stage1_feat - channel_mechanism_pred).abs(),
                    ],
                    dim=-1,
                )
            )
            mechanism_predictive_blend_weight = torch.sigmoid(self.mechanism_predictive_blend_logit)
            predictive_weight = mechanism_predictive_blend_weight
            if dual_expert_gate is not None:
                predictive_weight = dual_expert_gate * predictive_weight
            stage1_feat = stage1_feat + predictive_weight * mechanism_delta

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
        synthetic_rca_logits = None
        if self.synthetic_rca_head is not None:
            pooled = torch.cat(
                [
                    resid_enc.mean(dim=1),
                    resid_enc.amax(dim=1),
                ],
                dim=-1,
            )
            synthetic_rca_logits = self.synthetic_rca_head(pooled)

        # 步骤 6: 投影回 c_out
        resid_out = self.proj_out(resid_enc)  # (B, L, C)

        # 步骤 7: 趋势合并
        trend_out = self.trend_linear(trend)
        x_rec = resid_out + trend_out

        if self.use_source_interaction_head and return_root_score:
            interaction_recon_error = torch.abs(x_rec - x)
            if interaction_recon_error.shape[1] > 1:
                prev_interaction_error = torch.cat(
                    [
                        torch.zeros_like(interaction_recon_error[:, :1, :]),
                        interaction_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_interaction_error = torch.zeros_like(interaction_recon_error)
            interaction_onset_delta = torch.relu(
                interaction_recon_error - prev_interaction_error
            )
            interaction_mechanism_error = self._source_branch_mechanism_error(
                resid, A_adaptive, channel_mechanism_error
            )
            interaction_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            interaction_graph_delta = torch.abs(resid_adapted - resid)
            interaction_features = torch.stack(
                [
                    interaction_recon_error,
                    interaction_onset_delta,
                    interaction_mechanism_error,
                    interaction_gate,
                    interaction_graph_delta,
                ],
                dim=-1,
            )
            if self.training and self.source_interaction_detach_inputs:
                interaction_features = interaction_features.detach()
            source_interaction_logits, source_interaction_weights = (
                self.source_interaction_head(interaction_features)
            )

        if self.use_source_consistency_head and return_root_score:
            consistency_recon_error = torch.abs(x_rec - x)
            if consistency_recon_error.shape[1] > 1:
                prev_consistency_error = torch.cat(
                    [
                        torch.zeros_like(consistency_recon_error[:, :1, :]),
                        consistency_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_consistency_error = torch.zeros_like(consistency_recon_error)
            consistency_onset_delta = torch.relu(
                consistency_recon_error - prev_consistency_error
            )
            consistency_mechanism_error = self._source_branch_mechanism_error(
                resid, A_adaptive, channel_mechanism_error
            )
            consistency_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            consistency_gate_score = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.zeros_like(resid)
            )
            consistency_graph_delta = torch.abs(resid_adapted - resid)
            consistency_features = torch.stack(
                [
                    consistency_gate,
                    torch.log1p(consistency_onset_delta),
                    torch.log1p(torch.abs(consistency_mechanism_error)),
                    torch.log1p(consistency_recon_error),
                    torch.log1p(consistency_graph_delta),
                    torch.log1p(torch.clamp_min(consistency_gate_score, 0.0)),
                ],
                dim=-1,
            )
            if self.training and self.source_consistency_detach_inputs:
                consistency_features = consistency_features.detach()
            (
                source_consistency_logits,
                source_consistency_weights,
                source_consistency_support,
            ) = self.source_consistency_head(consistency_features)

        if (
            self.use_sps_role_head
            and self.sps_role_head is not None
            and strict_cross_self_error is not None
            and strict_cross_response_gain is not None
        ):
            if strict_cross_self_error.shape[1] > 1:
                previous_innovation = torch.cat(
                    [
                        torch.zeros_like(strict_cross_self_error[:, :1, :]),
                        strict_cross_self_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                previous_innovation = torch.zeros_like(strict_cross_self_error)
            strict_cross_onset = torch.relu(
                strict_cross_self_error - previous_innovation
            )
            sps_features = torch.stack(
                [
                    torch.log1p(strict_cross_self_error),
                    torch.log1p(strict_cross_response_gain),
                    torch.log1p(strict_cross_onset),
                    torch.log1p(resid.abs()),
                ],
                dim=-1,
            )
            if self.training and self.sps_role_detach_features:
                sps_features = sps_features.detach()
            sps_root_logits, sps_response_logits = self.sps_role_head(sps_features)
            # Existing export code consumes event responsibility logits.  The
            # value now comes from the jointly trained SPS role head instead
            # of a post-hoc source-gate score.
            event_responsibility_logits = sps_root_logits

        if self.use_event_responsibility_head:
            responsibility_mechanism_error = self._source_branch_mechanism_error(
                resid, A_adaptive, channel_mechanism_error
            )
            responsibility_causal_error = (
                causal_channel_error.to(dtype=resid.dtype)
                if causal_channel_error is not None
                else torch.zeros_like(resid)
            )
            responsibility_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            responsibility_gate_score = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.zeros_like(resid)
            )
            responsibility_recon_error = torch.abs(x_rec - x)
            if responsibility_recon_error.shape[1] > 1:
                prev_error = torch.cat(
                    [
                        torch.zeros_like(responsibility_recon_error[:, :1, :]),
                        responsibility_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_error = torch.zeros_like(responsibility_recon_error)
            responsibility_onset_delta = torch.relu(responsibility_recon_error - prev_error)
            responsibility_graph_delta = torch.abs(resid_adapted - resid)
            responsibility_features = torch.stack(
                [
                    torch.log1p(responsibility_recon_error),
                    torch.log1p(torch.abs(resid)),
                    torch.log1p(torch.abs(responsibility_mechanism_error)),
                    torch.log1p(torch.abs(responsibility_causal_error)),
                    responsibility_gate,
                    torch.log1p(torch.clamp_min(responsibility_gate_score, 0.0)),
                    torch.log1p(responsibility_onset_delta),
                    torch.log1p(responsibility_graph_delta),
                ],
                dim=-1,
            )
            if self.training and self.event_responsibility_detach_features:
                responsibility_features = responsibility_features.detach()
            event_responsibility_logits = self.event_responsibility_head(
                responsibility_features,
            )

        if self.use_response_suppressor_head and return_root_score:
            suppress_recon_error = torch.abs(x_rec - x)
            if suppress_recon_error.shape[1] > 1:
                prev_suppress_error = torch.cat(
                    [
                        torch.zeros_like(suppress_recon_error[:, :1, :]),
                        suppress_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_suppress_error = torch.zeros_like(suppress_recon_error)
            suppress_onset_delta = torch.relu(suppress_recon_error - prev_suppress_error)
            suppress_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            suppress_source_flow = suppress_gate * suppress_recon_error
            if self.use_channel_graph:
                suppress_propagation_support = self._graph_propagation_context(
                    suppress_source_flow,
                    A_adaptive,
                    target_evidence=suppress_recon_error,
                ).clamp_min(0.0)
            else:
                suppress_propagation_support = torch.zeros_like(suppress_recon_error)
            if root_response_propagation_support is None:
                root_response_propagation_support = suppress_propagation_support
            suppress_ratio = suppress_propagation_support / (
                suppress_recon_error + suppress_propagation_support + 1e-6
            )
            suppress_mechanism_error = self._source_branch_mechanism_error(
                resid,
                A_adaptive,
                channel_mechanism_error,
            )
            suppress_graph_delta = torch.abs(resid_adapted - resid)
            suppress_features = torch.stack(
                [
                    torch.log1p(suppress_recon_error),
                    torch.log1p(suppress_propagation_support),
                    suppress_ratio,
                    torch.log1p(suppress_graph_delta),
                    suppress_gate,
                    torch.log1p(suppress_onset_delta),
                    torch.log1p(torch.abs(suppress_mechanism_error)),
                ],
                dim=-1,
            )
            if self.training and self.response_suppressor_detach_inputs:
                suppress_features = suppress_features.detach()
            response_suppressor_logits = self.response_suppressor_head(
                suppress_features,
            )

        need_root_score_outputs = self.use_root_score_head and return_root_score
        if need_root_score_outputs:
            mechanism_error_feat = self._source_branch_mechanism_error(
                resid, A_adaptive, channel_mechanism_error
            )
            causal_error_feat = (
                causal_channel_error.to(dtype=resid.dtype)
                if causal_channel_error is not None
                else torch.zeros_like(resid)
            )
            gate_feat = source_gate if source_gate is not None else torch.zeros_like(resid)
            source_expert_prob = (
                torch.sigmoid(source_expert_logits)
                if source_expert_logits is not None
                else torch.zeros_like(resid)
            )
            graph_delta = torch.abs(resid_adapted - resid)
            stage_delta = torch.abs(stage1_feat - resid)
            recon_error = torch.abs(x_rec - x)
            if self.root_score_head_mode in {
                "root_response",
                "root-response",
                "response_decomposition",
            }:
                if recon_error.shape[1] > 1:
                    prev_recon_error = torch.cat(
                        [torch.zeros_like(recon_error[:, :1, :]), recon_error[:, :-1, :]],
                        dim=1,
                    )
                else:
                    prev_recon_error = torch.zeros_like(recon_error)
                onset_delta = torch.relu(recon_error - prev_recon_error)
                source_flow = gate_feat * recon_error
                if self.use_channel_graph:
                    propagation_support = self._graph_propagation_context(
                        source_flow,
                        A_adaptive,
                        target_evidence=recon_error,
                    ).clamp_min(0.0)
                else:
                    propagation_support = torch.zeros_like(recon_error)
                root_response_propagation_support = propagation_support
                if self.root_response_use_innovation_split:
                    source_innovation = torch.relu(recon_error - propagation_support)
                    explained_response = torch.minimum(recon_error, propagation_support)
                    source_features = torch.stack(
                        [
                            torch.log1p(recon_error),
                            torch.log1p(torch.abs(resid)),
                            torch.log1p(torch.abs(mechanism_error_feat)),
                            torch.log1p(torch.abs(causal_error_feat)),
                            gate_feat,
                            torch.log1p(stage_delta),
                            torch.log1p(onset_delta),
                            torch.log1p(source_flow),
                            torch.log1p(source_innovation),
                        ],
                        dim=-1,
                    )
                    response_terms = [
                        torch.log1p(explained_response),
                        torch.log1p(graph_delta),
                    ]
                else:
                    propagation_gap = torch.abs(recon_error - propagation_support)
                    source_features = torch.stack(
                        [
                            torch.log1p(recon_error),
                            torch.log1p(torch.abs(resid)),
                            torch.log1p(torch.abs(mechanism_error_feat)),
                            torch.log1p(torch.abs(causal_error_feat)),
                            gate_feat,
                            torch.log1p(stage_delta),
                            torch.log1p(onset_delta),
                            torch.log1p(source_flow),
                        ],
                        dim=-1,
                    )
                    response_terms = [
                        torch.log1p(propagation_support),
                        torch.log1p(graph_delta),
                        torch.log1p(propagation_gap),
                    ]
                if self.use_causal_response_evidence:
                    response_terms.append(
                        torch.log1p(
                            causal_response_support
                            if causal_response_support is not None
                            else torch.zeros_like(recon_error)
                        )
                    )
                response_features = torch.stack(response_terms, dim=-1)
                if self.use_source_expert_branch:
                    source_features = torch.cat(
                        [source_features, source_expert_prob.unsqueeze(-1)],
                        dim=-1,
                    )
                if self.training and self.root_score_detach_features:
                    source_features = source_features.detach()
                    response_features = response_features.detach()
                if self.use_pairwise_root_response_head:
                    (
                        root_score_logits,
                        root_response_source_evidence,
                        root_response_response_evidence,
                        root_response_source_confidence,
                        root_response_effective_response_evidence,
                        root_response_pairwise_relation,
                        root_response_pairwise_source_support,
                        root_response_pairwise_response_support,
                    ) = self.root_score_head(
                        source_features,
                        response_features,
                        A_adaptive=A_adaptive,
                    )
                else:
                    (
                        root_score_logits,
                        root_response_source_evidence,
                        root_response_response_evidence,
                        root_response_source_confidence,
                        root_response_effective_response_evidence,
                    ) = self.root_score_head(source_features, response_features)
            else:
                root_features = torch.stack(
                    [
                        torch.log1p(recon_error),
                        torch.log1p(torch.abs(resid)),
                        torch.log1p(torch.abs(mechanism_error_feat)),
                        torch.log1p(torch.abs(causal_error_feat)),
                        gate_feat,
                        torch.log1p(graph_delta),
                        torch.log1p(stage_delta),
                    ],
                    dim=-1,
                )
                if self.training and self.root_score_detach_features:
                    root_features = root_features.detach()
                root_score_logits = self.root_score_head(root_features).squeeze(-1)

        if self.use_evidence_fusion_head and return_root_score:
            fusion_recon_error = torch.abs(x_rec - x)
            if fusion_recon_error.shape[1] > 1:
                prev_fusion_error = torch.cat(
                    [
                        torch.zeros_like(fusion_recon_error[:, :1, :]),
                        fusion_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_fusion_error = torch.zeros_like(fusion_recon_error)
            fusion_onset_delta = torch.relu(fusion_recon_error - prev_fusion_error)
            fusion_mechanism_error = self._source_branch_mechanism_error(
                resid, A_adaptive, channel_mechanism_error
            )
            fusion_causal_error = (
                causal_channel_error.to(dtype=resid.dtype)
                if causal_channel_error is not None
                else torch.zeros_like(resid)
            )
            fusion_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            fusion_gate_score = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.zeros_like(resid)
            )
            fusion_root_prob = (
                torch.sigmoid(root_score_logits)
                if root_score_logits is not None
                else torch.zeros_like(resid)
            )
            fusion_event_prob = (
                torch.sigmoid(event_responsibility_logits)
                if event_responsibility_logits is not None
                else torch.zeros_like(resid)
            )
            evidence_terms = [
                torch.log1p(fusion_recon_error),
                torch.log1p(torch.abs(fusion_mechanism_error)),
                torch.log1p(torch.abs(fusion_causal_error)),
                fusion_gate,
                torch.log1p(torch.clamp_min(fusion_gate_score, 0.0)),
                torch.log1p(fusion_onset_delta),
                fusion_root_prob,
                fusion_event_prob,
            ]
            if self.evidence_fusion_use_graph_response:
                if (
                    root_response_propagation_support is not None
                    and root_response_propagation_support.shape
                    == fusion_recon_error.shape
                ):
                    fusion_propagation_support = (
                        root_response_propagation_support.to(dtype=resid.dtype)
                    )
                elif self.use_channel_graph:
                    fusion_propagation_support = self._graph_propagation_context(
                        fusion_gate * fusion_recon_error,
                        A_adaptive,
                        target_evidence=fusion_recon_error,
                    ).clamp_min(0.0)
                else:
                    fusion_propagation_support = torch.zeros_like(fusion_recon_error)
                fusion_unexplained = torch.relu(
                    fusion_recon_error - fusion_propagation_support
                )
                fusion_unexplained_ratio = fusion_unexplained / (
                    fusion_recon_error + fusion_propagation_support + 1e-6
                )
                evidence_terms.extend(
                    [
                        torch.log1p(fusion_unexplained),
                        fusion_unexplained_ratio,
                    ]
                )
            evidence_features = torch.stack(evidence_terms, dim=-1)
            if self.training and self.evidence_fusion_detach_inputs:
                evidence_features = evidence_features.detach()
            evidence_fusion_logits, evidence_fusion_weights = self.evidence_fusion_head(
                evidence_features,
            )

        if self.use_event_route_head and return_root_score:
            route_recon_error = torch.abs(x_rec - x)
            if route_recon_error.shape[1] > 1:
                prev_route_error = torch.cat(
                    [
                        torch.zeros_like(route_recon_error[:, :1, :]),
                        route_recon_error[:, :-1, :],
                    ],
                    dim=1,
                )
            else:
                prev_route_error = torch.zeros_like(route_recon_error)
            route_onset_delta = torch.relu(route_recon_error - prev_route_error)
            route_mechanism_error = self._source_branch_mechanism_error(
                resid,
                A_adaptive,
                channel_mechanism_error,
            )
            route_causal_error = (
                causal_channel_error.to(dtype=resid.dtype)
                if causal_channel_error is not None
                else torch.zeros_like(resid)
            )
            route_gate = source_gate if source_gate is not None else torch.zeros_like(resid)
            route_gate_score = (
                source_gate_score.to(dtype=resid.dtype)
                if source_gate_score is not None
                else torch.zeros_like(resid)
            )
            route_root_prob = (
                torch.sigmoid(root_score_logits)
                if root_score_logits is not None
                else torch.zeros_like(resid)
            )
            route_event_prob = (
                torch.softmax(event_responsibility_logits, dim=-1)
                if event_responsibility_logits is not None
                else torch.zeros_like(resid)
            )
            route_fusion_prob = (
                torch.sigmoid(evidence_fusion_logits)
                if evidence_fusion_logits is not None
                else torch.zeros_like(resid)
            )
            route_consistency_prob = (
                torch.sigmoid(source_consistency_logits)
                if source_consistency_logits is not None
                else torch.zeros_like(resid)
            )
            route_graph_delta = torch.abs(resid_adapted - resid)
            route_response_error = torch.abs(route_mechanism_error)
            if self.event_route_response_penalty > 0:
                event_route_graph_score = (
                    torch.log1p(route_recon_error)
                    + 0.55 * torch.log1p(torch.clamp_min(route_gate_score, 0.0))
                    + 0.35 * route_root_prob
                    + 0.35 * route_consistency_prob
                    + 0.15 * torch.log1p(route_graph_delta)
                )
                event_route_event_score = (
                    torch.log1p(route_recon_error)
                    + 0.65 * torch.log1p(route_onset_delta)
                    + 0.25 * route_fusion_prob
                    + 0.20 * route_event_prob
                    + 0.10 * torch.log1p(torch.abs(route_causal_error))
                )
                event_route_response_score = (
                    torch.log1p(route_response_error)
                    + 0.30 * route_event_prob
                    + 0.20 * torch.log1p(torch.abs(route_causal_error))
                    + 0.15 * torch.log1p(route_graph_delta)
                )
                route_response_feature = event_route_response_score
            else:
                event_route_graph_score = (
                    torch.log1p(route_recon_error)
                    + 0.50 * torch.log1p(route_response_error)
                    + 0.50 * torch.log1p(torch.clamp_min(route_gate_score, 0.0))
                    + 0.35 * route_root_prob
                    + 0.35 * route_consistency_prob
                )
                event_route_event_score = (
                    torch.log1p(route_recon_error)
                    + 0.60 * torch.log1p(route_onset_delta)
                    + 0.35 * route_event_prob
                    + 0.25 * route_fusion_prob
                    + 0.15 * torch.log1p(torch.abs(route_causal_error))
                )
                event_route_response_score = torch.zeros_like(event_route_graph_score)
                route_response_feature = route_response_error

            def _channel_entropy(values):
                channel_values = values.mean(dim=1).clamp_min(0.0)
                probs = torch.softmax(channel_values, dim=-1).clamp_min(1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1)
                return entropy / max(math.log(max(2, values.shape[-1])), 1e-8)

            def _channel_peak(values):
                channel_values = values.mean(dim=1).clamp_min(0.0)
                probs = torch.softmax(channel_values, dim=-1)
                return probs.max(dim=-1).values

            graph_entropy = _channel_entropy(event_route_graph_score)
            event_entropy = _channel_entropy(event_route_event_score)
            route_features = torch.stack(
                [
                    torch.log1p(event_route_graph_score.mean(dim=(1, 2))),
                    torch.log1p(event_route_event_score.mean(dim=(1, 2))),
                    _channel_peak(event_route_graph_score),
                    _channel_peak(event_route_event_score),
                    1.0 - graph_entropy,
                    1.0 - event_entropy,
                    route_gate.mean(dim=(1, 2)),
                    torch.log1p(route_response_feature.mean(dim=(1, 2))),
                    torch.log1p(route_graph_delta.mean(dim=(1, 2))),
                ],
                dim=-1,
            )
            if self.training and self.event_route_detach_inputs:
                route_features = route_features.detach()
                event_route_graph_score = event_route_graph_score.detach()
                event_route_event_score = event_route_event_score.detach()
                event_route_response_score = event_route_response_score.detach()
            event_route_logits, event_route_alpha = self.event_route_head(route_features)
            route_alpha = event_route_alpha.view(-1, 1, 1).to(dtype=resid.dtype)
            event_route_raw_score = (
                route_alpha * event_route_graph_score
                + (1.0 - route_alpha) * event_route_event_score
            )
            if self.event_route_response_penalty > 0:
                event_route_score = event_route_raw_score / (
                    1.0
                    + self.event_route_response_penalty
                    * event_route_response_score.clamp_min(0.0)
                )
            else:
                event_route_score = event_route_raw_score

        aux_losses = {}
        aux_losses['sparse_loss'] = self.get_sparse_loss()
        if hasattr(self.channel_graph, "get_dense_adjacency"):
            dense_channel_graph = self.channel_graph.get_dense_adjacency()
            if dense_channel_graph is not None:
                aux_losses['channel_graph_dense'] = dense_channel_graph
        aux_losses['vq_loss'] = vq_loss_val
        aux_losses['vq_dist'] = vq_dist  # (B, L) — 用于增强异常评分
        if (
            self.use_channel_corr_prior
            and self.lambda_channel_prior_align > 0
            and hasattr(self.channel_graph, "get_prior_align_loss")
        ):
            aux_losses['channel_prior_align_loss'] = self.channel_graph.get_prior_align_loss()
        if causal_score is not None:
            aux_losses['causal_score'] = causal_score
        if causal_channel_error is not None:
            aux_losses['causal_channel_error'] = causal_channel_error
        if causal_response_support is not None:
            aux_losses['causal_response_support'] = causal_response_support
        if causal_mechanism_loss is not None:
            aux_losses['causal_mechanism_loss'] = causal_mechanism_loss
            aux_losses['causal_sparse_loss'] = self.lagged_causal_graph.get_sparsity_loss()
        if strict_cross_full_error is not None:
            aux_losses['strict_cross_full_error'] = strict_cross_full_error
        if strict_cross_self_error is not None:
            aux_losses['strict_cross_self_error'] = strict_cross_self_error
        if strict_cross_response_gain is not None:
            aux_losses['strict_cross_response_gain'] = strict_cross_response_gain
        if strict_cross_mechanism_loss is not None:
            aux_losses['strict_cross_mechanism_loss'] = strict_cross_mechanism_loss
            aux_losses['strict_cross_sparse_loss'] = (
                self.strict_cross_mechanism.get_sparsity_loss()
            )
        if strict_cross_attention is not None:
            aux_losses['strict_cross_attention'] = strict_cross_attention
        if strict_cross_edge_weight is not None:
            aux_losses['strict_cross_edge_weight'] = strict_cross_edge_weight
        if self.strict_cross_mechanism is not None:
            aux_losses['strict_cross_prior_strength'] = torch.sigmoid(
                self.strict_cross_mechanism.prior_strength_raw
            )
        if channel_mechanism_score is not None:
            aux_losses['channel_mechanism_score'] = channel_mechanism_score
        if channel_mechanism_error is not None:
            aux_losses['channel_mechanism_error'] = channel_mechanism_error
            centered_resid = resid - resid.mean(dim=1, keepdim=True)
            aux_losses['channel_graph_target_variance'] = centered_resid.pow(2).mean(dim=1)
        if channel_mechanism_loss is not None:
            aux_losses['channel_mechanism_loss'] = channel_mechanism_loss
        if mechanism_coupling_weight is not None:
            aux_losses['mechanism_coupling_weight'] = mechanism_coupling_weight.detach()
        if source_preserving_weight is not None:
            aux_losses['source_preserving_weight'] = source_preserving_weight.detach()
            aux_losses['source_preserving_gate_mean'] = source_gate.detach().mean()
        if mechanism_feedback_weight is not None:
            aux_losses['mechanism_feedback_weight'] = mechanism_feedback_weight.detach()
        if mechanism_predictive_blend_weight is not None:
            aux_losses['mechanism_predictive_blend_weight'] = mechanism_predictive_blend_weight.detach()
        if corefinement_weight is not None:
            aux_losses['corefinement_weight'] = corefinement_weight.detach()
        if source_aware_corefinement_weight is not None:
            aux_losses['source_aware_corefinement_weight'] = (
                source_aware_corefinement_weight.detach()
            )
            aux_losses['source_aware_corefinement_gate_mean'] = source_gate.detach().mean()
        if source_gate is not None:
            aux_losses['source_gate'] = source_gate.detach()
            aux_losses['source_gate_prob'] = source_gate
            aux_losses['source_gate_score'] = source_gate_score
            aux_losses['source_gate_sparse_loss'] = source_gate.mean()
            if source_gate_onset_feature is not None:
                aux_losses['source_gate_onset_feature'] = source_gate_onset_feature
        if dual_expert_gate is not None:
            aux_losses['dual_expert_graph_gate'] = dual_expert_gate
            aux_losses['dual_expert_gate_mean'] = dual_expert_gate.detach().mean()
            aux_losses['dual_expert_independent_evidence'] = (
                dual_expert_independent_evidence
            )
            aux_losses['dual_expert_graph_evidence'] = dual_expert_graph_evidence
            aux_losses['dual_expert_feature_gap'] = (
                dual_expert_graph_feat - dual_expert_independent_feat
            ).abs()
        if source_expert_logits is not None:
            aux_losses['source_expert_logits'] = source_expert_logits
            aux_losses['source_expert_prob'] = torch.sigmoid(source_expert_logits)
            aux_losses['source_expert_graph_gate'] = source_expert_graph_gate
            aux_losses['source_expert_gate_mean'] = (
                source_expert_graph_gate.detach().mean()
            )
        if root_score_logits is not None:
            aux_losses['root_score_logits'] = root_score_logits
            aux_losses['root_score_prob'] = torch.sigmoid(root_score_logits)
        if root_response_source_evidence is not None:
            aux_losses['root_response_source_evidence'] = root_response_source_evidence
        if root_response_response_evidence is not None:
            aux_losses['root_response_response_evidence'] = root_response_response_evidence
        if root_response_source_confidence is not None:
            aux_losses['root_response_source_confidence'] = root_response_source_confidence
        if root_response_effective_response_evidence is not None:
            aux_losses['root_response_effective_response_evidence'] = (
                root_response_effective_response_evidence
            )
        if root_response_propagation_support is not None:
            aux_losses['root_response_propagation_support'] = (
                root_response_propagation_support
            )
        if root_response_pairwise_relation is not None:
            aux_losses['root_response_pairwise_relation'] = root_response_pairwise_relation
        if root_response_pairwise_source_support is not None:
            aux_losses['root_response_pairwise_source_support'] = (
                root_response_pairwise_source_support
            )
        if root_response_pairwise_response_support is not None:
            aux_losses['root_response_pairwise_response_support'] = (
                root_response_pairwise_response_support
            )
        if synthetic_logits is not None:
            aux_losses['synthetic_logits'] = synthetic_logits
        if synthetic_rca_logits is not None:
            aux_losses['synthetic_rca_logits'] = synthetic_rca_logits
        if event_responsibility_logits is not None:
            aux_losses['event_responsibility_logits'] = event_responsibility_logits
            aux_losses['event_responsibility_prob'] = torch.sigmoid(event_responsibility_logits)
        if sps_root_logits is not None:
            aux_losses['sps_root_logits'] = sps_root_logits
        if sps_response_logits is not None:
            aux_losses['sps_response_logits'] = sps_response_logits
        if event_route_logits is not None:
            aux_losses['event_route_logits'] = event_route_logits
            aux_losses['event_route_alpha'] = event_route_alpha
            aux_losses['event_route_score'] = event_route_score
            aux_losses['event_route_graph_score'] = event_route_graph_score
            aux_losses['event_route_event_score'] = event_route_event_score
            aux_losses['event_route_response_score'] = event_route_response_score
        if response_suppressor_logits is not None:
            aux_losses['response_suppressor_logits'] = response_suppressor_logits
            aux_losses['response_suppressor_prob'] = torch.sigmoid(
                response_suppressor_logits
            )
        if evidence_fusion_logits is not None:
            aux_losses['evidence_fusion_logits'] = evidence_fusion_logits
            aux_losses['evidence_fusion_prob'] = torch.sigmoid(evidence_fusion_logits)
            aux_losses['evidence_fusion_responsibility'] = torch.softmax(
                evidence_fusion_logits,
                dim=-1,
            )
        if evidence_fusion_weights is not None:
            aux_losses['evidence_fusion_weights'] = evidence_fusion_weights
        if source_interaction_logits is not None:
            aux_losses['source_interaction_logits'] = source_interaction_logits
            aux_losses['source_interaction_prob'] = torch.sigmoid(source_interaction_logits)
        if source_interaction_weights is not None:
            aux_losses['source_interaction_weights'] = source_interaction_weights.detach()
        if source_consistency_logits is not None:
            aux_losses['source_consistency_logits'] = source_consistency_logits
            aux_losses['source_consistency_prob'] = torch.sigmoid(source_consistency_logits)
        if source_consistency_weights is not None:
            aux_losses['source_consistency_weights'] = source_consistency_weights.detach()
        if source_consistency_support is not None:
            aux_losses['source_consistency_support'] = source_consistency_support
        if self._last_graph_propagation_lag_weights is not None:
            aux_losses['graph_propagation_lag_weights'] = (
                self._last_graph_propagation_lag_weights
            )
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
        channel_mechanism_score_dict = {}
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
            if self.use_channel_mechanism_score and aux_losses.get('channel_mechanism_score', None) is not None:
                m_score = aux_losses['channel_mechanism_score']
                channel_mechanism_score_dict[ws] = m_score[:, :ws] if ws < self.win_size else m_score
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

        if self.use_channel_mechanism_score and channel_mechanism_score_dict:
            mechanism_score = self._aggregate_vq_dist(
                channel_mechanism_score_dict, valid_sizes, B, L, x.device,
            )
            mechanism_score = self._normalize_channel_mechanism_score(mechanism_score)
            mechanism_score = mechanism_score[:, -score.shape[1]:] if mechanism_score.shape[1] >= score.shape[1] else mechanism_score
            if mechanism_score.shape[1] < score.shape[1]:
                pad_len = score.shape[1] - mechanism_score.shape[1]
                pad_mech = torch.zeros(B, pad_len, device=score.device)
                mechanism_score = torch.cat([pad_mech, mechanism_score], dim=1)
            score = score + self.channel_mechanism_score_weight * mechanism_score

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
