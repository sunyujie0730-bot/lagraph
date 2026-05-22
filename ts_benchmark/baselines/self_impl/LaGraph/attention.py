"""
LaGraph 注意力机制模块
这个模块实现了带有通道掩码（Channel Mask）的时间序列注意力机制。

与普通Transformer注意力的区别：
- 普通Attention：每个位置可以关注所有其他位置（全连接）
- OrdAttention：通过可学习的通道掩码，动态决定哪些时间步之间可以互相"看到"
  
为什么要用通道掩码？
- 不是所有时间步都应该互相影响
- 通过学习哪些时间步相关，可以让模型更聚焦于重要的依赖关系
- 类似于让模型自己学习一个"稀疏注意力"模式
"""

import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from .channel_mask import channel_mask_generator


class OrdAttention(nn.Module):
    """
    带通道掩码的多头注意力（OrdAttention = Ordered Attention）
    
    名字由来："Ord" 指 Ordered（有序的），因为时间序列是有顺序的，
    这个注意力专门为有序的时间序列数据设计。
    
    核心流程：
    1. 输入经过深度可分离卷积（dwconv）提取局部时间特征
    2. 切分为 Q、K、V 三个矩阵
    3. 计算注意力分数 scores = Q * K^T / sqrt(d)
    4. 用通道掩码过滤掉不重要的时间步对
    5. 加权求和得到输出
    
    参数字典：
        win_size:  窗口大小（时间步数）
        model_dim: 模型隐藏维度
        atten_dim: 每个注意力头的维度
        head_num:  注意力头的数量（多头注意力可以关注不同的模式）
        dropout:   dropout比例
        residual:  是否使用残差连接
    """
    def __init__(self, win_size, model_dim, atten_dim, head_num, dropout, residual):
        super(OrdAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual
        self.win_size = win_size
        
        # Q、K、V 的投影矩阵：将 model_dim 映射到 atten_dim * head_num
        # 为什么是 atten_dim * head_num？因为每个头有独立的 Q、K、V
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        # 1x1卷积：生成 QKV 的初始表示（跨时间步的信息混合）
        self.qkv = nn.Conv1d(self.win_size, self.win_size * 3, kernel_size=1, bias=True)
        # 深度可分离卷积：在每个时间步周围取3个邻居，提取局部模式
        # groups参数表示每个通道独立卷积，大大减少参数量
        self.qkv_dwconv = nn.Conv1d(
            self.win_size * 3, self.win_size * 3,
            kernel_size=3, stride=1, padding=1,
            groups=self.win_size * 3, bias=True
        )
        
        # 最终的全连接层：将多头结果合并并映射回 model_dim
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        # LayerNorm：对每个样本的特征维度做归一化，帮助训练稳定
        self.norm = nn.LayerNorm(model_dim)
        # 通道掩码生成器：学习哪些时间步之间应该有关联
        self.mask = channel_mask_generator(self.atten_dim, self.win_size)

    def forward(self, Q, K, V):
        """
        前向传播（虽然参数叫Q、K、V，但LaGraph中Q=K=V=x，即自注意力）
        
        输入形状：(B, L, D)  ——> 批次、时间步数、特征维度
        输出形状：(B, L, D)  ——> 同样形状，但融入了注意力信息
        """
        # 保存输入用于残差连接
        residual = Q.clone()
        
        # --- 步骤1: 生成QKV ---
        # 先通过1x1卷积将L映射到3L（相当于同时生成Q、K、V的门控信号）
        # 再通过深度可分离卷积捕捉局部时间模式
        qkv = self.qkv_dwconv(self.qkv(Q))
        # 沿第1维切成3份：Q、K、V
        Q, K, V = qkv.chunk(3, dim=1)
        
        # --- 步骤2: 线性投影 + 多头拆分 ---
        # 将维度映射到 atten_dim * head_num，然后拆成 (B, L, head_num, atten_dim)
        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        # --- 步骤3: 调整维度顺序 ---
        # 变成 (B, head_num, L, atten_dim)，方便批量矩阵乘法
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # --- 步骤4: 计算注意力分数 ---
        # scores[b, h, i, j] = Q[b,h,i] · K[b,h,j] / sqrt(d)
        # 除以sqrt(d)是为了防止点积过大导致softmax梯度消失
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        
        # --- 步骤5: 应用通道掩码 ---
        mask = self.mask(K)  # 生成二值掩码矩阵 (B, head_num, L, L)
        # 对于mask=0的位置，赋一个极大负数，这样softmax后会变成0
        large_negative = -math.log(1e10)
        attention_mask = torch.where(mask == 0, large_negative, 0)
        # mask乘法和加法结合：mask可以让某些位置收缩，attention_mask直接屏蔽
        scores = scores * mask + attention_mask
        
        # --- 步骤6: Softmax + 加权求和 ---
        attn = nn.Softmax(dim=-1)(scores)  # 将分数转为概率分布
        context = torch.matmul(attn, V)     # 用注意力权重加权V
        
        # --- 步骤7: 恢复原始形状 ---
        context = context.transpose(1, 2)  # (B, L, head_num, atten_dim)
        context = context.reshape(residual.size(0), residual.size(1), -1)  # 展平多头
        
        # --- 步骤8: 最终投影 + Dropout ---
        output = self.dropout(self.fc(context))

        # 残差连接：把原始输入加回去
        # 这解决了深层网络中的梯度消失问题，让模型更容易训练
        if self.residual:
            return self.norm(output + residual), attn
        else:
            return self.norm(output), attn
