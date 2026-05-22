# -*- coding: utf-8 -*-
"""
LaGraph v11.2 Encoder 堆叠模块：精简时序编码器
=============================================

根据最终精简路线（2026-05-22）:

★ 保留: Dynamic Scale Selection（根因 4 — 动态尺度缺失）
  问题: MultiScaleTemporalConv 使用全局静态 softmax 权重融合 4 个 dilation，
        SWaT 突变需小 dilation，MSL 漂移需大 dilation，同一组权重无法同时满足
  方案: 替换为 SE-style 动态门控，根据输入特征自适应选择尺度权重

v9-v11.1 历史:
  v9: 初始 EncoderStack + MultiScaleTemporalConv + OrdAttention + FFN
  v9.6: FFN gradient checkpointing (后移除)
  v10: d_model=128, atten_dim=16, 移除 checkpointing
  v11.1: 修复 FFN 前向兼容性

每层结构 (v11.2，带动态尺度):
  ┌──────────────────────────────────┐
  │  输入 x (B, L, d_model)          │
  │       ↓                         │
  │  MultiScaleTemporalConv (动态门控) │
  │       ↓                         │
  │  OrdAttention (时序自注意力)       │
  │       ↓                         │
  │  FFN (无 gradient checkpointing)  │
  │       ↓                         │
  │  输出 x                          │
  └──────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import OrdAttention


# ══════════════════════════════════════════════════════════════════
#  EncoderStack
# ══════════════════════════════════════════════════════════════════


class EncoderStack(nn.Module):
    """
    多层时序编码器 (v11.2)

    每层包含:
    1. 多尺度时序卷积（MultiScaleTemporalConv with 动态门控）
    2. 时序自注意力（OrdAttention）
    3. FFN

    v11.2 变更:
    - MultiScaleTemporalConv 使用 SE-style 动态门控（P0 Dynamic Scale Selection）
    - forward 不再返回 freq_features

    v9 变更：
    - 移除 freq_branch（频域增强分支，移至 gcn_model.FreqTower1D）
    - 移除 cross_channel（跨通道注意力，与 ChannelAdaptiveGraph 重叠）
    """

    def __init__(self, num_layers, d_model, d_ff, win_size, n_heads,
                 d_state, d_conv, expand, dropout,
                 use_freq_branch=False, use_cross_channel=False,
                 use_multi_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, d_ff=d_ff, win_size=win_size,
                n_heads=n_heads, dropout=dropout,
                use_multi_scale=use_multi_scale,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────
#  EncoderLayer
# ──────────────────────────────────────────────────────────────


class EncoderLayer(nn.Module):
    """
    单层时序编码器 (v11.2)

    v11.2: MultiScaleTemporalConv 使用动态门控（SE-style）
    """

    def __init__(self, d_model, d_ff, win_size, n_heads, dropout,
                 use_multi_scale=True):
        super().__init__()

        self.use_multi_scale = use_multi_scale
        self.ffn_dropout_p = dropout

        # ---- 多尺度时序卷积 (★ v11.2: 动态门控版) ----
        if use_multi_scale:
            self.multi_scale = MultiScaleTemporalConv(
                d_model=d_model, dropout=dropout
            )
            self.norm_ms = nn.LayerNorm(d_model)

        # ---- 时序自注意力 (OrdAttention) ----
        _atten_dim = max(8, d_model // 8)  # d_model=128 → 16
        self.attention = OrdAttention(
            win_size=win_size, model_dim=d_model, atten_dim=_atten_dim,
            head_num=n_heads, dropout=dropout, residual=True
        )
        self.norm_attn = nn.LayerNorm(d_model)

        # ---- FFN ----
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
        self.norm_ffn = nn.LayerNorm(d_model)

    def _ffn_forward(self, x):
        h = F.linear(x, self.ffn1.weight, self.ffn1.bias)
        h = F.gelu(h)
        h = F.dropout(h, p=self.ffn_dropout_p, training=self.training)
        h = F.linear(h, self.ffn2.weight, self.ffn2.bias)
        h = x + h
        h = F.layer_norm(h, normalized_shape=h.shape[-1:],
                          weight=self.norm_ffn.weight,
                          bias=self.norm_ffn.bias)
        return h

    def forward(self, x):
        # ---- 1. 多尺度时序卷积 (★ v11.2: 动态门控) ----
        if self.use_multi_scale:
            x_ms = self.multi_scale(x)
            x = self.norm_ms(x + x_ms)

        # ---- 2. 时序自注意力 ----
        x_attn, _ = self.attention(x, x, x)
        x = self.norm_attn(x + x_attn)

        # ---- 3. FFN ----
        x = self._ffn_forward(x)

        return x


# ══════════════════════════════════════════════════════════════════
#  ★ P0 [MODIFIED] MultiScaleTemporalConv — 动态尺度选择
# ══════════════════════════════════════════════════════════════════

class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时序卷积 (v11.2 Dynamic Scale Selection)

    ★ P0 修改: 替换静态 softmax 权重为 SE-style 动态门控

    旧版:
      self.scale_weights = nn.Parameter(torch.ones(4)/4)
      weights = F.softmax(self.scale_weights, dim=0)  ← 全局静态
      merged = sum(w * o for w, o in zip(weights, outputs))

    新版:
      1. 对输入做全局平均池化，得到 context vector (B, d_model)
      2. 通过两层 MLP (reduction=4) 生成 4 个动态权重
      3. 权重 = softmax(动态权重) — 每个样本独立的尺度融合

    尺度:
    - dilation=1:  细粒度短期依赖（SWaT 突变）
    - dilation=3:  中粒度依赖
    - dilation=7:  粗粒度长期依赖
    - dilation=15: 超长期趋势（MSL 漂移）

    参数量增加: ~128×32 + 32×4 ≈ 4.2K（可忽略）
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        dilations = [1, 3, 7, 15]
        kernel_size = 3
        self.num_scales = len(dilations)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model, kernel_size=kernel_size,
                dilation=d, padding=d,
                groups=d_model // max(1, min(d_model, 8)),
            )
            for d in dilations
        ])

        # ★ P0: SE-style 动态尺度门控
        #   squeeze: 全局平均池化 (B, d_model, L) → (B, d_model, 1)
        #   excitation: FC-ReLU-FC-Softmax → 4 个动态权重
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, self.num_scales),
        )

        self.merge = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # 保留静态权重作为初始化参考（仅用于 warmup 阶段）
        self.static_weights = nn.Parameter(
            torch.ones(self.num_scales) / self.num_scales,
            requires_grad=False,  # 不训练，仅作为回退
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model) — 输入

        Returns:
            merged: (B, L, d_model) — 动态融合后的输出
        """
        x_conv = x.permute(0, 2, 1)  # (B, d_model, L)

        # 步骤1: 各尺度卷积
        outputs = []
        for conv in self.convs:
            out = conv(x_conv)
            out = out[:, :, :x.shape[1]]  # trim to original length
            outputs.append(out)

        # ★ P0: 步骤2: SE-style 动态权重生成
        #   squeeze: (B, d_model, L) → (B, d_model, 1)
        context = self.squeeze(x_conv).squeeze(-1)  # (B, d_model)
        #   excitation: (B, d_model) → (B, num_scales)
        scale_logits = self.excitation(context)  # (B, 4)
        dynamic_weights = F.softmax(scale_logits, dim=-1)  # (B, 4)

        # 步骤3: 动态加权融合
        #   outputs: list of (B, d_model, L)
        #   dynamic_weights: (B, 4)
        merged = sum(
            w.unsqueeze(-1).unsqueeze(-1) * o
            for w, o in zip(dynamic_weights.unbind(dim=-1), outputs)
        )  # (B, d_model, L)

        merged = merged.permute(0, 2, 1)  # (B, L, d_model)
        merged = self.activation(self.merge(merged))
        return self.dropout(merged)


# 保持向后兼容的别名
GraphStack = EncoderStack
