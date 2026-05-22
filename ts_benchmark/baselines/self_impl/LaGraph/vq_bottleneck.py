# -*- coding: utf-8 -*-
"""
VQ Bottleneck: 向量量化瓶颈层 —— Dual-Path 版本 (v11.2)

★ P0 修改: Dual-Path VQ 重构（根因 3 — VQ 能力受限）

根因:
  原实现 VQ 是"串行"瓶颈: 特征 → VQ量化 → 量化特征 → EncoderStack
  问题: 量化导致的离散化信息丢失会传递到后续编码器，降低重建质量

Dual-Path 方案:
  输入特征
    ├── 主路径 (Continuous Path): 特征直接通过，保持连续信息流
    │   (无信息损失，供 EncoderStack 重建使用)
    └── 旁路 VQ (VQ Bypass): 特征经 VQ 量化后计算 VQ 距离
        (VQ 距离作为异常评分特征，不参与重建)

  输出:
    - vq_dist: (B, L) — 各位置到最近 codebook 的 VQ 距离（用于异常评分）
    - vq_loss: scalar — commitment loss（用于训练）
    - continuous_feat: (B, L, C) — 原始输入特征（主路径，无量化损失）

参考: VQ-VAE (van den Oord et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQBottleneck(nn.Module):
    """
    向量量化瓶颈层 —— Dual-Path (v11.2)

    支持两种模式:
      - serial_mode=True (默认): 旧版串行 VQ，前向返回量化特征 (兼容旧代码)
      - serial_mode=False (推荐): Dual-Path，旁路 VQ 仅计算距离，主路径直通

    Args:
        dim:             特征维度
        codebook_size:   codebook 大小
        commitment_cost:  commitment loss 权重
        serial_mode:     是否为串行模式 (True=兼容旧版, False=Dual-Path)
    """

    def __init__(self, dim, codebook_size=64, commitment_cost=0.25, serial_mode=False):
        super(VQBottleneck, self).__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.serial_mode = serial_mode

        self.codebook = nn.Parameter(
            torch.randn(codebook_size, dim) * 0.01
        )

        # ★ Dual-Path: VQ 距离 → 可学习尺度缩放（用于异常评分中的归一化）
        self.vq_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z):
        """
        Args:
            z: (B, L, C) — 输入特征

        Returns:
            serial_mode=True (兼容旧版):
                z_q:      (B, L, C) — 量化后的特征
                vq_loss:  scalar — commitment loss
                vq_dist:  (B, L) — 每个位置到最近 codebook 的距离
            serial_mode=False (Dual-Path, 推荐):
                continuous_feat:  (B, L, C) — 主路径连续特征 (即 z 本身)
                vq_loss:         scalar — commitment loss
                vq_dist:         (B, L) — 每个位置到最近 codebook 的距离
        """
        B, L, C = z.shape

        # Flatten for VQ
        z_flat = z.reshape(-1, C)  # (B*L, C)

        # 计算 z_flat 和 codebook 之间的距离
        z_sq = (z_flat ** 2).sum(dim=-1, keepdim=True)        # (B*L, 1)
        e_sq = (self.codebook ** 2).sum(dim=-1).unsqueeze(0)  # (1, K)
        ze = torch.matmul(z_flat, self.codebook.T)            # (B*L, K)
        dist = z_sq + e_sq - 2 * ze                            # (B*L, K)
        dist = torch.clamp(dist, min=1e-10)

        # 最近 codebook 索引
        min_idx = torch.argmin(dist, dim=-1)  # (B*L,)

        # 量化
        z_q_flat = self.codebook[min_idx]    # (B*L, C)

        # Commitment loss：鼓励编码器输出接近 codebook
        vq_loss_commit = F.mse_loss(z_flat, z_q_flat.detach())
        # Codebook loss：鼓励 codebook 接近编码器输出
        vq_loss_codebook = F.mse_loss(z_flat.detach(), z_q_flat)
        vq_loss = vq_loss_commit * self.commitment_cost + vq_loss_codebook

        # VQ 偏离度：每个位置到最近 codebook 的距离
        min_dist = dist[
            torch.arange(dist.shape[0], device=dist.device), min_idx
        ]  # (B*L,)
        vq_dist = min_dist.reshape(B, L)  # (B, L)

        # ★ Dual-Path Decision
        if self.serial_mode:
            # 旧版串行模式: Straight-through estimator
            # z_q = z + (z_q_flat - z_flat).detach()
            z_q_flat = z_flat + (z_q_flat - z_flat).detach()
            z_q = z_q_flat.reshape(B, L, C)
            return z_q, vq_loss, vq_dist
        else:
            # ★ Dual-Path 模式 (v11.2):
            #   主路径 = 原始连续特征 z（无量化损失）
            #   旁路 VQ 距离通过可学习缩放归一化，用于异常评分
            return z, vq_loss, vq_dist
