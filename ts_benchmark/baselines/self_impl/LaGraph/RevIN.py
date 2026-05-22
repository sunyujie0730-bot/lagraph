"""
RevIN — 可逆实例归一化（Reversible Instance Normalization）
一种专为时间序列设计的归一化方法，常用于异常检测模型。

问题背景：
  普通的BatchNorm/LayerNorm会"抹掉"数据的统计信息（均值和方差）
  但异常检测恰恰需要这些统计信息来判断异常！

RevIN 的解决方案：
  1. 先记录原始数据的均值和标准差（保存"身份信息"）
  2. 做归一化让模型训练更稳定
  3. 在输出时用保存的统计信息"还原"回去
  4. 这样既能享受归一化的训练好处，又不会丢失关键信息

类比理解：
  就像拍照时用滤镜（归一化让画面更好看），
  但拍完后把原始光线信息（均值和方差）写进元数据，
  需要时可以从元数据还原真实场景（反归一化）。
"""

import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    可逆实例归一化
    
    工作流程（两步）：
    ┌─────────────────────────────────┐
    │ mode='norm'（训练/推理时）        │
    │   x → 计算均值/方差并保存          │
    │     → 减均值除以标准差            │
    │     → 可选：仿射变换               │
    │                                   │
    │ mode='denorm'（输出还原时）        │
    │   x → 可选：逆仿射变换             │
    │     → 乘标准差加均值              │
    │     → 还原到原始数据尺度           │
    └─────────────────────────────────┘
    
    参数说明：
        num_features: 特征/通道数（变量个数）
        eps:          数值稳定性参数，防止除以0（默认1e-5）
        affine:       是否学习仿射变换参数（可训练的缩放和平移）
                      类似于BatchNorm的 γ 和 β
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps          # 防止方差为0时除以0
        self.affine = affine    # 是否学习额外的缩放和平移
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        前向传播，根据 mode 参数执行不同操作
        
        mode='norm':   归一化模式 — 输入时使用
        mode='denorm': 反归一化模式 — 输出时使用，还原数据
        """
        if mode == 'norm':
            # 第1步：记录当前batch的统计信息（均值、标准差）
            self._get_statistics(x)
            # 第2步：用这些统计信息做归一化
            x = self._normalize(x)
        elif mode == 'denorm':
            # 用之前记录的统计信息反归一化
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """
        初始化仿射参数（可学习的 γ 和 β）
        
        affine_weight (γ): 初始化为全1，即不做缩放
        affine_bias (β):   初始化为全0，即不做平移
        
        形状都是 (C,)，每个通道（变量）有自己独立的缩放和平移
        """
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        # 自动选择设备（GPU或CPU）
        self.affine_weight = self.affine_weight.to(
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.affine_bias = self.affine_bias.to(
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )

    def _get_statistics(self, x):
        """
        计算并保存输入数据的统计信息
        
        计算范围：所有非批次、非特征维度
        例如输入 (B, L, C)，则对 L 维度求均值和方差
        
        .detach() 防止梯度通过统计信息回传（保持统计值稳定）
        """
        # 确定要归约的维度：排除第0维(批次)和最后一维(特征)
        dim2reduce = tuple(range(1, x.ndim - 1))
        # 均值：例如对时间维度求平均
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # 标准差：方差 + eps 再开方（eps防止方差为0）
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """
        归一化操作：z-score标准化
        
        公式：x_norm = (x - mean) / stdev
        可选：x_norm = γ * x_norm + β（仿射变换）
        
        效果：让数据变成均值≈0、方差≈1的分布，利于模型训练
        """
        x = x - self.mean          # 减均值 → 中心化
        x = x / self.stdev          # 除以标准差 → 缩放到单位方差
        if self.affine:
            x = x * self.affine_weight  # γ缩放：让模型学习最佳方差
            x = x + self.affine_bias     # β平移：让模型学习最佳均值
        return x

    def _denormalize(self, x):
        """
        反归一化操作：将归一化后的数据还原到原始尺度
        
        公式（无仿射）：x_orig = x * stdev + mean
        公式（有仿射）：先逆仿射，再逆标准化
        
        为什么需要反归一化？
        - 模型输出是归一化后的值
        - 计算重建误差需要和原始数据对比
        - 所以必须还原到原始数据的尺度
        """
        if self.affine:
            # 逆仿射变换：(x - β) / γ
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)  # eps*eps 防止除以0
        # 逆标准化：x * stdev + mean
        x = x * self.stdev
        x = x + self.mean
        return x
