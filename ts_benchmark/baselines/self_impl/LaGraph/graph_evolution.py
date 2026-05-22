# -*- coding: utf-8 -*-
"""
DCGN 图演化正则化层 (Neural ODE 风格)
=====================================

对因果邻接矩阵施加平滑演化约束，防止图结构突变。

核心思想：
  dA/dt = f_θ(A, x)
  正常工况下因果图随特征变化平滑演化，图结构突变即为异常信号。

实现方式：
  - EMA 跟踪邻接矩阵的跨 batch 平滑参考值 A_ref
  - 小型神经网络 f_θ 根据特征 x 预测"允许的"图调整量 δ
  - A_allowed = A_ref + δ
  - 损失: L_smooth = MSE(A_current, A_allowed)
    当图突变 > 特征变化能解释的范围时，损失增大 → 检测"图级异常"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEvolutionLayer(nn.Module):
    """
    图演化正则化器

    对因果邻接矩阵施加时序平滑约束。维持一个 EMA 参考图 A_ref，
    用小型 ODE 网络根据当前特征预测允许的图变化范围，
    惩罚超出该范围的图突变。

    参数：
        num_nodes:  节点数量 (= 通道数 C)
        hidden_dim: ODE 网络隐藏层维度
        ema_decay:  EMA 衰减系数 (越接近 1 越平滑)
    """

    def __init__(self, num_nodes, hidden_dim=64, ema_decay=0.99):
        super(GraphEvolutionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        # ODE 速度网络: f(A_ref_flat || x_summary) → delta_flat
        # 输入 = C*C (图展平) + C (特征时间均值) = C² + C
        in_dim = num_nodes * num_nodes + num_nodes
        self.ode_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_nodes * num_nodes),
        )

        # EMA 参考邻接矩阵 (C, C)，跨 batch 平滑追踪
        self.register_buffer('A_ref', torch.zeros(num_nodes, num_nodes))
        self._initialized = False

        # 缓存最近一次前向传播的 ODE 损失
        self._ode_loss = torch.tensor(0.0)

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def forward(self, A_current, x):
        """
        前向传播：对邻接矩阵施加平滑约束（透传 A_current，不修改）

        参数：
            A_current: (B, C, C) 当前因果邻接矩阵
            x:         (B, L, C) 当前窗口特征

        返回：
            A_current: (B, C, C) 原样返回，仅作为占位符保持管道兼容
        """
        B, C, _ = A_current.shape
        assert C == self.num_nodes, (
            f"GraphEvolutionLayer: C={C} != num_nodes={self.num_nodes}"
        )

        # -- 更新 EMA 参考图 --
        with torch.no_grad():
            A_mean = A_current.detach().mean(0)  # (C, C)
            if not self._initialized:
                self.A_ref.copy_(A_mean)
                self._initialized = True
            else:
                self.A_ref.mul_(self.ema_decay).add_(
                    A_mean, alpha=1.0 - self.ema_decay
                )

        # -- 扩展参考图到 batch 维度 --
        A_ref_exp = self.A_ref.unsqueeze(0).expand(B, -1, -1)  # (B, C, C)

        # -- 特征时间摘要 --
        x_summary = x.mean(dim=1)  # (B, C)

        # -- ODE 网络预测允许的图调整量 --
        A_flat = A_ref_exp.reshape(B, C * C)                  # (B, C²)
        ode_in = torch.cat([A_flat, x_summary], dim=-1)       # (B, C²+C)
        delta_flat = self.ode_net(ode_in)                     # (B, C²)
        delta = delta_flat.reshape(B, C, C)                   # (B, C, C)

        # A_allowed = 参考图 + 特征驱动的微调
        A_allowed = A_ref_exp + delta
        A_allowed = F.softmax(A_allowed, dim=-1)

        # 平滑损失：当前图 vs 允许图之间的均方误差
        # 用 detach 防止梯度通过 A_allowed 回流影响 ODE 预测目标
        self._ode_loss = F.mse_loss(A_current, A_allowed.detach())

        return A_current

    def get_ode_loss(self):
        """获取最近一次前向传播的图演化平滑损失标量"""
        return self._ode_loss
