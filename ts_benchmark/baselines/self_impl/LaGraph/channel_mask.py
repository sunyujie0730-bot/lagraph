"""
LaGraph 通道掩码生成器
这个模块生成一个可学习的二值掩码矩阵，用于控制注意力机制中
哪些时间步之间可以互相"看到"。

核心思想：
- 不是所有时间步对都应该被关注（全连接注意力太密集）
- 通过学习一个掩码矩阵，动态筛选出重要的时间步对
- 使用 STE (Straight-Through Estimator) 技巧让二值化操作可微分

类比理解：
  想象你在看一部电影，不是每一帧都和所有其他帧有关。
  通道掩码就像学会了"快进到关键帧"，只关注真正的关键时刻。
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax


class channel_mask_generator(torch.nn.Module):
    """
    通道掩码生成器：学习一个稀疏的注意力掩码
    
    工作流程：
    1. 输入 K 矩阵 (B, head_num, L, atten_dim)
    2. 通过简单的线性层+Sigmoid生成 (0,1) 之间的分布矩阵
    3. 用 STE 技巧将连续值二值化为 {0, 1}
    4. 对角线强制为1（每个时间步至少要能"看到"自己）
    
    参数字典：
        input_size: 输入维度（注意力头的维度）
        n_vars:     时间步数（窗口大小 L）
    """
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        # 简单的生成器：线性投影 → Sigmoid压缩到(0,1)
        # 输入是 atten_dim 维，输出是 n_vars 维（每个时间步一个门控值）
        self.generator = nn.Sequential(
            torch.nn.Linear(input_size, n_vars, bias=False),
            nn.Sigmoid()  # 压缩到(0,1)区间，代表"该时间步被允许参与注意力的概率"
        )
        # 初始化为全0权重，让模型从最保守的策略开始学习
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        """
        前向传播
        
        输入: K 矩阵 (B, head_num, L, atten_dim)
        输出: 掩码矩阵 (B, head_num, L, L)，每个元素 ∈ [0, 1]
        """
        # --- 步骤1: 生成分布矩阵 ---
        # 对每个位置，生成一个 (0,1) 的概率值
        # 形状: (B, head_num, L, L) — 每对时间步有一个"相关性分数"
        distribution_matrix = self.generator(x)
        
        # --- 步骤2: 用STE二值化 ---
        # 将 (0,1) 的概率 → 硬二值化 (0 或 1)
        # STE 技巧让这个不可微的操作在反向传播时能传递梯度
        resample_matrix = self._ste_resample(distribution_matrix)
        
        # --- 步骤3: 处理对角线 ---
        # 构造反单位矩阵（对角线为0，其余为1）
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        # 单位矩阵（对角线为1）
        diag = torch.eye(self.n_vars).to(x.device)
        
        # 把非对角线位置用掩码过滤，然后强制对角线为1
        # 这确保每个时间步至少会关注自己（self-attention的基础要求）
        resample_matrix = torch.einsum("bchd,dd->bchd", resample_matrix, inverse_eye) + diag
        
        return resample_matrix

    def _ste_resample(self, distribution_matrix):
        """
        STE (直通估计器) 二值化技巧
        
        问题：>0.5 这个比较操作是不可微的，无法反向传播梯度
        解决：前向传播用硬二值，反向传播用连续值近似
        
        原理公式：
        forward: mask = (logits > 0.5).float()   ← 硬二值
        backward: ste_mask = mask + (logits - logits.detach())  ← 梯度流过logits
        
        简化理解：
        - 前向：像开关一样，>0.5就是1，<=0.5就是0
        - 反向：假装这个开关是连续的，让梯度能传回去
        """
        b, c, h, d = distribution_matrix.shape
        logits = distribution_matrix  # 已经经过 Sigmoid，值在 (0,1) 之间

        # 硬二值化：大于0.5的变成1，否则变成0
        mask = (logits > 0.5).float()
        # STE 核心：mask - logits 的梯度被 detach() 截断
        # 所以反向传播时梯度直接流向 logits（不走 mask 分支）
        ste_mask = (mask - logits).detach() + logits
        return ste_mask
