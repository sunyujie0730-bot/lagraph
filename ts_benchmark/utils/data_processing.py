# -*- coding: utf-8 -*-
"""
数据处理工具模块

提供通用的数据切分、清洗等操作。

目前包含：
  - split_before: 按指定位置切分时间序列数据（最常用的操作）
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd


def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    在指定位置切分时间序列数据为两段

    这个函数在整个项目中大量使用，用于：
      - 划分训练集/测试集（前面80%训练，后面20%测试）
      - 切出验证集
      - 按时间点分段分析

    支持两种数据类型：
      - pandas DataFrame：按行切片，保留列名和索引
      - NumPy ndarray：按[行, 列]切片

    用法示例：
        train_data, test_data = split_before(df, 8000)
        # df共10000行 → train: 前8000行, test: 后2000行

    参数：
        data:  时间序列数据（DataFrame 或 ndarray）
        index: 切分的行索引位置（前index行归第一部分）
    返回：
        (前半段, 后半段) 的元组
    """
    if isinstance(data, pd.DataFrame):
        # DataFrame：用 iloc 切分
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        # NumPy 数组：直接用数组切片
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")
