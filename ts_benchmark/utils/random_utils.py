# -*- coding: utf-8 -*-
"""
随机种子工具模块

负责固定所有随机数生成器的种子，
确保实验结果可复现。

为什么需要固定随机种子？
  机器学习中到处都有随机性：
  - 参数初始化是随机的
  - 数据打乱是随机的（shuffle）
  - Dropout 是随机的
  - cuDNN 的卷积算法选择也可能不确定

  不固定种子 → 每次运行结果可能不同 → 无法复现实验。
  固定种子后 → 相同的代码 + 相同的种子 = 完全相同的结果。

两种固定方式：
  - fix_random_seed: 基本版，只固定 Python/PyTorch/NumPy
  - fix_all_random_seed: 完全版，还固定 CUDA 和 cuDNN 行为
"""

import os
import random
from typing import Optional, NoReturn

import numpy as np
import torch


def fix_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    """
    固定基本随机种子（轻量版）

    只固定三个最核心的随机源：
    1. Python 内置的 random 模块
    2. PyTorch 的随机数生成器
    3. NumPy 的随机数生成器

    适用场景：
    - 纯 CPU 实验
    - 快速调试，不需要完全确定性
    - GPU 环境但不在意 cuDNN 的非确定性

    注意：这不固定 CUDA 相关的随机性，GPU 上结果可能略有波动。

    参数：
        seed: 种子值，默认 2021。传 None 表示跳过（不固定）。
    """
    if seed is None:
        return

    random.seed(seed)       # Python 自带随机库
    torch.manual_seed(seed) # PyTorch 随机数（CPU）
    np.random.seed(seed)    # NumPy 随机数


def fix_all_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    """
    固定所有随机种子（完全版）

    这是推荐的实验用种子函数，几乎消除所有随机性来源：

    固定的内容包括：
    1. PyTorch CPU 随机数
    2. PyTorch 所有 GPU 的随机数
    3. NumPy 随机数
    4. Python 随机数
    5. cuDNN 行为：强制使用确定性算法
       - deterministic = True：只用确定性的卷积算法
       - benchmark = False：不自动搜索最快算法（搜索过程也有随机性）
    6. Python hash seed 环境变量

    为什么还要固定 PYTHONHASHSEED？
      Python 中 dict/set 的哈希值默认是随机的（安全考虑）。
      这会影响某些迭代顺序，进而影响结果。
      固定为 '1' 后，每次运行哈希顺序一致。

    注意：完全确定性会牺牲一些 GPU 性能，
    因为 cuDNN 的确定性算法通常比启发式算法慢。

    参数：
        seed: 种子值，默认 2021。传 None 表示跳过。
    """
    if seed is None:
        return

    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed_all(seed)    # PyTorch 所有 GPU
    np.random.seed(seed)                # NumPy
    random.seed(seed)                   # Python

    # 强制 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True  # 只用确定性的卷积实现
    torch.backends.cudnn.benchmark = False     # 不自动搜索最快算法

    # 再次固定 CUDA 种子（双保险）
    torch.cuda.manual_seed(seed)

    # 固定 Python 的哈希种子（确保 dict/set 迭代顺序一致）
    os.environ['PYTHONHASHSEED'] = str(1)
