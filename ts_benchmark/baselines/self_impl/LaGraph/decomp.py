"""
LaGraph 序列分解模块
将时间序列分解为两个部分：
- 趋势分量（Trend）：数据的长期、平滑变化趋势（如股价的长期上涨走势）
- 残差分量（Residual）：去除趋势后的剩余部分（如股价的短期波动）

为什么这样做？
- 趋势是平滑的，不需要复杂的图结构来处理
- 残差包含高频波动和异常信号，需要通过GCN+Attention深度处理
- 分开处理可以让模型更专注，提高效率

类比：就像把一张照片分离为"背景"（趋势）和"前景细节"（残差）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    移动平均块：通过滑动平均提取序列的平滑趋势
    
    工作原理：
    - 用一个滑动窗口（如25个时间步）取平均值
    - 相当于"低通滤波器"，过滤掉高频波动，只保留低频趋势
    
    类比：像用毛玻璃看时间序列，细节变模糊了，但整体趋势更清晰了
    
    参数：
        kernel_size: 平滑窗口大小，越大趋势越平滑（但也可能丢失细节）
        stride:      滑动步长
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 一维平均池化 = 移动平均
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        输入 x: (B, L, C) — 批次、时间步、通道数
        输出:   (B, L, C) — 平滑后的序列（相同形状）
        """
        # 在序列两端做"对称填充"，避免边界效应
        # 例如 kernel_size=25，两端各填充12个值（用首尾值复制）
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # AvgPool1d 需要 (B, C, L) 格式，所以先转置
        x = self.avg(x.permute(0, 2, 1))
        # 再转回来 (B, L, C)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    标准序列分解块（固定窗口=25的移动平均）
    
    分解公式：
        残差 = 原始序列 - 移动平均（趋势）
    
    这是最简单常用的分解方式，MoEDecomposition 是它的升级版
    """

    def __init__(self):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(25, stride=1)  # 固定25步窗口

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 提取趋势
        res = x - moving_mean             # 残差 = 原始 - 趋势
        return res, moving_mean


class MovingAvg(nn.Module):
    """
    轻量级移动平均（仅用于 MoE 专家内部）
    与 moving_avg 功能相同，但代码更简洁
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class GatingNet(nn.Module):
    """
    门控网络：决定每个"专家"的重要性权重
    
    作用：不同的时间序列片段可能适合不同的平滑窗口
    - 剧烈波动时用小窗口（5步）更好
    - 平稳时用大窗口（45步）更好
    - 门控网络学会根据输入自动分配权重
    
    类比：像一个"投票机制"，根据当前情况决定听哪个专家的意见更多
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNet, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # 全局平均池化：把整个序列压缩成一个标量
            nn.Flatten(),                # 展平
            nn.Linear(input_dim, num_experts),  # 全连接：输出每个专家的权重
        )

    def forward(self, x):
        """
        输入: x (B, L, C)
        输出: weights (B, num_experts) — 每个样本对每个专家的权重，和为1
        """
        # 转置为 (B, C, L) 用于池化（池化在时间维度上）
        x = x.permute(0, 2, 1)
        weights = F.softmax(self.gate(x), dim=1)  # softmax确保权重和为1
        return weights


class MoEDecomposition(nn.Module):
    """
    MoE 序列分解器（MoE = Mixture of Experts，混合专家）
    
    这是标准序列分解的升级版：
    - 使用多个不同窗口大小的移动平均作为"专家"
    - 通过门控网络学习如何组合这些专家的输出
    
    专家列表（窗口大小）：
    [5, 15, 25, 35, 45] — 从小窗口到大窗口
    - 小窗口(5)：捕捉快速变化的中期趋势
    - 大窗口(45)：捕捉非常平滑的长期趋势
    - 多个窗口组合：自适应地在精细和粗粒度之间平衡
    
    类比：就像同时用尺子（小窗口）和卷尺（大窗口）量东西，
    然后根据测量对象决定更相信哪个工具的结果
    """
    def __init__(self, input_dim, kernel_sizes=[5, 15, 25, 35, 45]):
        super(MoEDecomposition, self).__init__()
        # 创建多个移动平均"专家"，每个有不同的窗口大小
        self.experts = nn.ModuleList([MovingAvg(k) for k in kernel_sizes])
        # 门控网络：学会根据输入分配每个专家的权重
        self.gating = GatingNet(input_dim=input_dim, num_experts=len(kernel_sizes))

    def forward(self, x):
        """
        输入: x (B, L, C)
        输出:
            res:         残差分量 (B, L, C) — 用GCN+Attention处理
            moving_mean: 趋势分量 (B, L, C) — 用轻量卷积处理
        """
        # --- 步骤1: 每个专家各自做移动平均 ---
        # 得到5个不同平滑程度的趋势估计
        expert_outputs = [expert(x) for expert in self.experts]  # 5个 (B, L, C)
        
        # --- 步骤2: 堆叠专家输出 ---
        stacked_means = torch.stack(expert_outputs, dim=1)  # (B, 5, L, C)
        
        # --- 步骤3: 门控网络计算权重 ---
        # 权重是基于输入x自动学习的
        gate_weights = self.gating(x).unsqueeze(-1).unsqueeze(-1)  # (B, 5, 1, 1)
        
        # --- 步骤4: 加权求和 ---
        # 最终趋势 = w1*专家1 + w2*专家2 + ... + w5*专家5
        # 不同时间、不同样本可能有不同的权重分配
        moving_mean = torch.sum(gate_weights * stacked_means, dim=1)  # (B, L, C)
        
        # --- 步骤5: 残差 = 原始 - 趋势 ---
        res = x - moving_mean
        
        return res, moving_mean
