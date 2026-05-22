# -*- coding: utf-8 -*-
"""
评估器模块（Evaluator）
======================
负责计算模型的各项评估指标，是整个基准测试中"打分"的核心组件。

核心功能：
  1. 根据配置动态加载多个评估指标（acc, precision, recall, f1, VUS等）
  2. 用模型预测结果和真实标签计算分数
  3. 处理评估过程中的异常（某个指标计算失败不影响其他指标）

类比理解：
  就像考试的"批卷老师"：
  - 拿到标准答案（actual真实标签）和考生答案（predicted模型预测）
  - 按评分标准（metric指标配置）逐项打分
  - 每道题打错了（计算异常）就标记为 NaN，继续批下一道
"""
import functools
import traceback
from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import METRICS


def encode_params(params):
    """
    将评估指标的参数编码为字符串（方便展示）
    
    例如：{"threshold": 0.5, "alpha": 0.01}
    编码为："alpha:0.01;threshold:0.5"（按字母排序）
    
    浮点数会保留3位小数，避免过长的字符串
    
    参数：params 指标参数字典
    返回：编码后的参数字符串
    """
    encoded_pairs = []
    for key, value in sorted(params.items()):
        if isinstance(value, (np.floating, float)):
            value = round(value, 3)
        encoded_pairs.append(f"{key}:{repr(value)}")
    return ";".join(encoded_pairs)


class Evaluator:
    """
    评估器：管理评估指标并执行评分
    
    工作流程：
    1. 初始化时根据配置的指标列表加载对应的计算函数
    2. 调用 evaluate() 用真实值和预测值逐一计算各指标
    3. 返回所有指标的评分结果列表
    
    为什么指标用 functools.partial？
    - 同一个指标可能用不同参数跑多次（如 F1 的 threshold 不同）
    - partial 把参数"预绑定"到函数上，调用时只要传数据即可
    - 例如：METRICS["f1"] 是一个函数，partial(f1, threshold=0.5) 生成带阈值的版本
    """

    def __init__(self, metric: List[dict]):
        """
        初始化评估器
        
        参数：metric 指标配置列表，每个元素是一个字典
              格式：{"name": "指标名", "参数1": 值1, "参数2": 值2}
              如：[{"name": "f1", "threshold": 0.5}, {"name": "precision"}]
        
        初始化过程：
        1. 遍历每个指标配置
        2. 从 METRICS 注册表中找到对应的计算函数
        3. 如果有额外参数，用 partial 绑定
        4. 生成带参数的指标全名（如 "f1;threshold:0.5"）
        """
        self.metric = metric
        self.metric_funcs = []  # 已绑定参数的计算函数列表
        self.metric_names = []  # 指标名称列表（含参数信息）

        # 遍历每个指标配置，创建对应的计算函数
        for metric_info in self.metric:
            # 拷贝一份，避免修改原始配置
            metric_info_copy = metric_info.copy()
            # 取出指标名称
            metric_name = metric_info_copy.pop("name")
            # 如果还有剩余参数（如 threshold），追加到名称后面
            if metric_info_copy:
                metric_name += ";" + encode_params(metric_info_copy)
            self.metric_names.append(metric_name)
            
            # 创建计算函数
            metric_name_copy = metric_info.copy()
            name = metric_name_copy.pop("name")
            fun = METRICS[name]  # 从注册表中获取原始函数
            if metric_name_copy:
                # 有额外参数，用 partial 绑定
                self.metric_funcs.append(functools.partial(fun, **metric_name_copy))
            else:
                # 无额外参数，直接用原始函数
                self.metric_funcs.append(fun)

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: Union[np.ndarray, pd.DataFrame] = None,
        **kwargs,
    ) -> list:
        """
        计算模型的各项评估指标
        
        参数：
            actual:    真实标签/值（ground truth），形状如 (时间步, 通道数)
            predicted: 模型预测结果，形状如 (时间步, 通道数)
            scaler:    归一化器对象（某些指标需要还原数据后再评估）
            hist_data: 历史数据（某些指标需要历史上下文，如VUS）
        
        返回：指标值列表，与 metric_funcs 一一对应
        
        数据预处理：
        - 3D → 2D：如果输入是3维张量（批次, 时间, 通道），先展平
        - 这使后续指标计算统一处理2维数据
        """

        # 将3维输入展平为2维（合并批次和时间维度）
        if actual.ndim == 3:
            actual = actual.reshape(-1, actual.shape[1])
        if predicted.ndim == 3:
            predicted = predicted.reshape(-1, predicted.shape[1])

        # 处理历史数据：统一转为 numpy 数组
        if isinstance(hist_data, pd.DataFrame):
            hist_data_np = hist_data.values
        else:
            hist_data_np = hist_data.reshape(-1, hist_data.shape[1])

        # 依次调用每个指标函数，收集结果
        return [
            m(actual, predicted, scaler=scaler, hist_data=hist_data_np)
            for m in self.metric_funcs
        ]

    def evaluate_with_log(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: np.ndarray = None,
        **kwargs,
    ) -> Tuple[List[Any], str]:
        """
        计算评估指标并记录错误日志（比 evaluate() 更健壮）
        
        与 evaluate() 的区别：
        - evaluate() 如果某个指标计算出错，整个流程崩溃
        - evaluate_with_log() 捕获异常，标记为 NaN 并记录日志，继续计算下一个
        
        这在批量评估时非常重要：
        不会因为某个数据上某指标算不了就中断整个实验
        
        返回：(指标值列表, 错误日志字符串)
        """
        evaluate_result = []
        log_info = ""
        for m in self.metric_funcs:
            try:
                evaluate_result.append(
                    m(actual, predicted, scaler=scaler, hist_data=hist_data)
                )
            except Exception as e:
                # 记录异常但仍继续（关键设计！）
                evaluate_result.append(np.nan)
                log_info += f"Error in calculating {m.__name__}: {traceback.format_exc()}\n{e}\n"
        return evaluate_result, log_info

    def default_result(self):
        """
        返回默认的评估结果（全 NaN）
        
        用途：当模型评估完全失败时，返回全NaN作为兜底值
              避免因为缺少结果导致报告生成报错
        
        返回：与指标数量相同长度的 NaN 列表
        """
        return len(self.metric_names) * [np.nan]
