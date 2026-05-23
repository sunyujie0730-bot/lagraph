# -*- coding: utf-8 -*-
"""
LaGraph v11.3 图模块：双图协同（邻近性调制 + Top-K 稀疏化）
=============================================================

v11.3 术语修正（根据学术投稿要求）:

1. ★ ChannelAdaptiveGraph: 单层消息传递 — 实例级自适应通道依赖
   - Top-K 稀疏化 + 对称化（§3.2.3）
   - L1 正则化促进稀疏性

2. ★ SimplifiedTemporalGraph: 时序邻近性调制（§4.2）
   - forward 新增参数 A_proximity: (B, C, C) 通道邻近性矩阵
   - 通过 A_proximity 调制时序注意力温度参数
   - 强局部结构 -> 更锐利的时序注意力（选择性增强）
   - 邻近性语义注入时序建模

3. ★ 保留 A_sym 对称化（§3.2.3）+ Top-K 稀疏化 + L1 正则

底层工具: nconv, linear, GCN — 图卷积基础算子（不变）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  底层工具（不变）
# ══════════════════════════════════════════════════════════════════

class nconv(nn.Module):
    """图卷积算子：X' = X · A"""
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bfn,bnv->bfv', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """1x1 卷积实现特征维度变换"""
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv1d(c_in, c_out, kernel_size=1,
                                   padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    """多层图卷积：聚合多跳邻居信息"""
    def __init__(self, c_in, c_out, dropout, order=3):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in_total = (order + 1) * c_in
        self.mlp = linear(c_in_total, c_out)
        self.dropout = nn.Dropout(dropout)
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.relu(h)
        return h


# ══════════════════════════════════════════════════════════════════
#  ChannelAdaptiveGraph: §2.3 自适应通道图（对称化 + 两层消息传递）
# ══════════════════════════════════════════════════════════════════

class ChannelAdaptiveGraph(nn.Module):
    """
    数据驱动的自适应通道依赖图（v11.3: 术语修正 + Top-K 稀疏化）

    v11.3 变更:
    - 术语替换: causal_topk -> sparse_topk
    - Top-K 稀疏化 + 对称化（§3.2.3）
    - L1 正则化促进稀疏性

    输入:  x  (B, L, C)
    输出:  x_out (B, L, C), A (B, C, C)

    参数:
        num_nodes:    通道数 C
        topk:         嵌入维度
        sparse_topk: top-k 稀疏化
        dropout:      dropout
    """

    def __init__(self, num_nodes, topk=5,
                 sparse_topk=None, dropout=0.1):
        super(ChannelAdaptiveGraph, self).__init__()
        self.num_nodes = num_nodes
        self.nodedim = topk

        # === 先验图嵌入（学习性结构）===
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, topk) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(topk, num_nodes) * 0.1)

        # === 数据驱动的通道编码器 ===
        self.channel_encoder = nn.Sequential(
            nn.Linear(num_nodes, num_nodes * 2),
            nn.LayerNorm(num_nodes * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_nodes * 2, num_nodes),
        )

        # === 每样本门控：数据驱动成分的强度 ===
        self.gate_net = nn.Sequential(
            nn.Linear(num_nodes, num_nodes // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_nodes // 2, 1),
            nn.Sigmoid(),
        )

        # === 数据调制系数（可学习）===
        self.gamma = nn.Parameter(torch.tensor(0.3))

        # === 温度参数（softmax 温度调节）===
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # === Top-k 稀疏化 ===
        self.sparse_topk = sparse_topk
        if sparse_topk is None or sparse_topk >= num_nodes:
            self.effective_topk = num_nodes
        else:
            self.effective_topk = sparse_topk

        # === L1 正则缓存 ===
        self._l1_penalty = torch.tensor(0.0)
        self._warmup_alpha = 0.0

    def set_warmup_progress(self, alpha: float):
        self._warmup_alpha = alpha

    def get_l1_penalty(self) -> torch.Tensor:
        return self._l1_penalty * self._warmup_alpha

    def forward(self, x):
        """
        x: (B, L, C)

        返回: x_out (B, L, C), A (B, C, C)
              A 为对称化 + Top-K 稀疏化后的邻接矩阵
        """
        B, L, C = x.shape
        assert C == self.num_nodes, f"ChannelAdaptiveGraph: input C={C} != num_nodes={self.num_nodes}"

        # ---- 1. 先验嵌入 ----
        nv1 = self.nodevec1.unsqueeze(0).expand(B, -1, -1)  # (B, C, topk)
        nv2 = self.nodevec2.unsqueeze(0).expand(B, -1, -1)  # (B, topk, C)
        A_base = torch.bmm(nv1, nv2)  # (B, C, C)

        # ---- 2. 数据驱动的成对交互 ----
        x_mean = x.mean(dim=1)  # (B, C)
        x_feat = self.channel_encoder(x_mean)  # (B, C)

        # 成对交互: outer product
        A_data = torch.bmm(x_feat.unsqueeze(-1), x_feat.unsqueeze(1))  # (B, C, C)
        A_data = A_data / (C ** 0.5 + 1e-8)
        A_data = torch.tanh(A_data)  # 约束到 [-1, 1]

        # ---- 3. 每样本自适应门控 ----
        gate_val = self.gate_net(x_mean).unsqueeze(-1)  # (B, 1, 1)

        # ---- 4. 融合：先验 + 数据驱动 ----
        A_combined = A_base / self.temperature.clamp(min=0.1) + self.gamma * A_data * gate_val

        # ---- 5. 对称化处理（无向图语义）----
        A_logit = A_combined
        A_sym_logit = (A_logit + A_logit.transpose(-2, -1)) / 2.0
        A_sym = F.softmax(A_sym_logit, dim=-1)
        A_sym = torch.nan_to_num(A_sym, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 6. Top-k 稀疏化 ----
        if self.effective_topk < C:
            _, topk_idx = torch.topk(A_sym, k=self.effective_topk, dim=-1)
            mask = torch.zeros_like(A_sym).scatter_(-1, topk_idx, 1.0)
            A_sym = A_sym * mask
            row_sum = A_sym.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            A_sym = A_sym / row_sum

        A = A_sym  # (B, C, C)

        # ---- 7. 单层消息传递（v11.1: 修复过度平滑）----
        x_out = torch.bmm(x, A) + x  # (B, L, C)

        # ---- 8. L1 稀疏正则化 ----
        self._l1_penalty = A.abs().mean()

        return x_out, A


# ══════════════════════════════════════════════════════════════════
#  SimplifiedTemporalGraph: §3.1.4 简化时序模块（v11.3 邻近性调制）
# ══════════════════════════════════════════════════════════════════

class SimplifiedTemporalGraph(nn.Module):
    """
    简化时序图（v11.3: 术语替换 — 时序邻近性调制）

    v11.3 变更:
    - 参数名 A_causal -> A_proximity
    - 内部模块 causal_encoder -> proximity_temp_encoder
    - 注释中"因果"全部替换为"邻近性"/"locality"
    - forward 参数 A_proximity: (B, C, C) 通道邻近性矩阵

    架构:
        1. 可学习位置编码 -> 时序邻接矩阵 A_temp (LxL)
        2. A_temp Laplacian 平滑 -> 时序图卷积
        3. 多尺度时序卷积（局部时间模式）
        4. 邻近性调制温度（v11.3 术语修正）
        5. 残差连接 + LayerNorm

    输入:  x (B, L, C) — 通道图混合后的特征
           A_proximity (B, C, C) — 通道邻近性矩阵（默认 None，使用自我注意力）
    输出:  x_out (B, L, C), A_temp (B, L, L)

    参数:
        win_size:   L 时间窗口
        d_model:    C 特征维度
        dropout:    dropout rate
    """

    def __init__(self, win_size, d_model, dropout=0.1):
        super(SimplifiedTemporalGraph, self).__init__()
        self.win_size = win_size
        self.d_model = d_model

        # === 可学习位置编码（时序邻接矩阵基础）===
        self.pos_embed = nn.Parameter(torch.randn(win_size, max(16, win_size // 4)) * 0.02)
        self.temp_scale = nn.Parameter(torch.tensor(1.0))

        # === 邻近性温度编码器（v11.3: causal_encoder -> proximity_temp_encoder）===
        # 将 (B, C, C) 邻近性矩阵映射为 (B, 1) 温度调制强度
        # 1. unsqueeze(1) -> (B, 1, C, C) 为 4D
        # 2. AdaptiveAvgPool2d -> (B, 1, 1, 1) 池化
        # 3. Flatten -> (B, 1) -> MLP 映射为标量强度
        self.proximity_temp_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # === 多尺度时序卷积（局部时间模式）===
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model)
            for k in [3, 5, 7]
        ])
        self.conv_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # === 全局时间注意力（单头，轻量）===
        self.temp_attn = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=True,
        )

        # === 输出投影 ===
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # === 门控：数据自适应调制温度 ===
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, A_proximity=None):
        """
        x: (B, L, C) — 通道图混合后的特征
        A_proximity: (B, C, C) 或 None — 通道邻近性矩阵（v11.3 参数名修正）

        返回: x_out (B, L, C), A_temp (B, L, L)
        """
        B, L, C = x.shape

        # ---- 1. 时序邻接矩阵（可学习位置编码）----
        pos_sim = self.pos_embed @ self.pos_embed.T  # (L, L)

        # ---- 1b. v11.3 邻近性偏置：根据局部结构强度调制温度 ----
        if A_proximity is not None:
            # 从邻近性图提取全局局部结构强度
            proximity_strength = self.proximity_temp_encoder(A_proximity.unsqueeze(1))  # (B, 1)
            # proximity_strength: 强局部结构 -> 接近 1 -> 温度更低 -> 注意力更锐利
            # temp: [0.8, 2.0] 范围
            temp_min, temp_max = 0.5, 2.0
            temp = temp_min + (temp_max - temp_min) * (1.0 - proximity_strength.squeeze(-1))
        else:
            temp = self.temp_scale.clamp(min=0.1, max=10.0).expand(B)

        # 逐样本温度
        A_temp_raw = pos_sim.unsqueeze(0).expand(B, -1, -1) / temp.unsqueeze(-1).unsqueeze(-1).clamp(min=0.1)
        A_temp = F.softmax(A_temp_raw, dim=-1)  # (B, L, L)

        # ---- 2. 时序图卷积（Laplacian 平滑）----
        D = A_temp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        D_sqrt_inv = torch.rsqrt(D)
        A_norm = A_temp * D_sqrt_inv * D_sqrt_inv.transpose(-1, -2)
        x_gcn = torch.bmm(A_norm, x)  # (B, L, C)

        # ---- 3. 多尺度时序卷积 ----
        x_t = x.transpose(1, 2)  # (B, C, L)
        conv_feats = [conv(x_t) for conv in self.convs]  # 各 (B, C, L)
        conv_feat = torch.cat([f.transpose(1, 2) for f in conv_feats], dim=-1)  # (B, L, 3C)
        x_conv = self.conv_fusion(conv_feat)  # (B, L, C)

        # ---- 4. 全局时间注意力 ----
        gate = self.gate_net(x_t).unsqueeze(-1)  # (B, 1, 1)
        x_attn, _ = self.temp_attn(x, x, x, need_weights=False)

        # ---- 5. 融合：平滑 + 卷积 + 注意力 ----
        x_fused = x_gcn + x_conv + gate * x_attn
        x_fused = self.norm(x_fused)
        x_out = x + self.dropout(x_fused)  # 残差

        return x_out, A_temp


class DynamicTemporalGraph(nn.Module):
    """
    Content-adaptive temporal graph.

    Unlike SimplifiedTemporalGraph, the temporal adjacency is generated from the
    current window by QK attention, then sparsified by row-wise top-k. A small
    positional prior and local-distance bias keep the graph stable early in
    training, while the content term carries the dynamic event-specific signal.
    """

    def __init__(self, win_size, d_model, dropout=0.1, attn_dim=None,
                 temporal_topk=None, local_radius=None):
        super(DynamicTemporalGraph, self).__init__()
        self.win_size = win_size
        self.d_model = d_model
        self.attn_dim = attn_dim or min(64, max(16, d_model))
        self.temporal_topk = temporal_topk or max(8, win_size // 4)
        self.local_radius = float(local_radius or max(4, win_size // 10))

        self.q_proj = nn.Linear(d_model, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.pos_embed = nn.Parameter(
            torch.randn(win_size, max(16, win_size // 4)) * 0.02
        )
        self.content_scale = nn.Parameter(torch.tensor(1.0))
        self.pos_scale = nn.Parameter(torch.tensor(0.2))
        self.local_scale = nn.Parameter(torch.tensor(0.5))

        self.proximity_temp_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
            for k in [3, 5, 7]
        ])
        self.conv_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        idx = torch.arange(win_size)
        dist = (idx[:, None] - idx[None, :]).abs().float()
        self.register_buffer("time_distance", dist, persistent=False)

    def forward(self, x, A_proximity=None):
        B, L, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        content_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.attn_dim)

        pos = self.pos_embed[:L]
        pos_logits = pos @ pos.T
        local_bias = -self.time_distance[:L, :L].to(x.device) / self.local_radius

        logits = (
            self.content_scale.clamp(min=0.1, max=5.0) * content_logits
            + self.pos_scale.clamp(min=0.0, max=2.0) * pos_logits.unsqueeze(0)
            + self.local_scale.clamp(min=0.0, max=5.0) * local_bias.unsqueeze(0)
        )

        if A_proximity is not None:
            proximity_strength = self.proximity_temp_encoder(A_proximity.unsqueeze(1))
            temp = 0.6 + 1.4 * (1.0 - proximity_strength.squeeze(-1))
        else:
            temp = x.new_ones(B)
        logits = logits / temp.view(B, 1, 1).clamp(min=0.2)

        k_keep = min(self.temporal_topk, L)
        if k_keep < L:
            topk_idx = torch.topk(logits, k=k_keep, dim=-1).indices
            sparse_logits = logits.new_full(logits.shape, -1e4)
            logits = sparse_logits.scatter(-1, topk_idx, logits.gather(-1, topk_idx))

        A_temp = F.softmax(logits, dim=-1)
        A_temp = torch.nan_to_num(A_temp, nan=0.0, posinf=0.0, neginf=0.0)
        x_dyn = self.out_proj(torch.bmm(A_temp, v))

        x_t = x.transpose(1, 2)
        conv_feats = [conv(x_t) for conv in self.convs]
        conv_feat = torch.cat([feat.transpose(1, 2) for feat in conv_feats], dim=-1)
        x_conv = self.conv_fusion(conv_feat)

        gate = self.fusion_gate(x_t).unsqueeze(-1)
        x_fused = gate * x_dyn + (1.0 - gate) * x_conv
        x_fused = self.norm(x_fused)
        x_out = x + self.dropout(x_fused)

        return x_out, A_temp
