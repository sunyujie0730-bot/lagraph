# -*- coding: utf-8 -*-
"""
LaGraph 基准测试流水线（Pipeline）
=================================
这是整个 LaGraph 框架的"总调度中心"，负责按顺序协调所有的基准测试流程。

整体流程（4大步骤）：
  步骤1: 加载配置 → 确定用哪些数据集、模型、评估策略
  步骤2: 准备数据 → 从本地加载CSV文件，放入数据服务器
  步骤3: 构建模型 → 根据配置创建所有待评估的模型实例
  步骤4: 评估模型 → 每个模型在每个数据集上跑一遍，收集结果
  步骤5: 保存日志 → 把评估结果写入CSV文件，方便后续分析

类比理解：
  就像工厂流水线：
  - 配置 = 生产计划单（今天要生产什么）
  - 数据 = 原材料（CSV文件）
  - 模型 = 加工机器（不同的异常检测算法）
  - 评估 = 质检环节（给每个机器的结果打分）
  - 日志 = 质检报告（记录所有分数）
"""
from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import List, Dict, Type, Optional

import pandas as pd

from ts_benchmark.data.data_source import (
    LocalForecastingDataSource,
    LocalStForecastingDataSource,
    DataSource,
    LocalAnomalyDetectDataSource,
)
from ts_benchmark.data.suites.global_storage import GlobalStorageDataServer
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models import get_models
from ts_benchmark.recording import save_log
from ts_benchmark.utils.parallel import ParallelBackend


@dataclass
class DatasetInfo:
    """
    数据集元信息
    
    定义了每个预定义数据集的属性：
    - size_value: 该数据集允许的大小级别（large/small/user）
    - datasrc_class: 该数据集使用的数据源类
    
    比如 "large_detect" 允许 large 和 small 两种大小，
    使用 LocalAnomalyDetectDataSource 来加载数据
    """
    # 允许的 size 字段值列表
    size_value: List
    # 对应的数据源类
    datasrc_class: Type[DataSource]


# ============================================================
# 预定义数据集注册表
# 通过名称快速获取数据集的配置信息
# ============================================================
PREDEFINED_DATASETS = {
    # --- 预测类数据集 ---
    "large_forecast": DatasetInfo(
        size_value=["large", "small"],
        datasrc_class=LocalForecastingDataSource,
    ),
    "small_forecast": DatasetInfo(
        size_value=["small"], datasrc_class=LocalForecastingDataSource
    ),
    "user_forecast": DatasetInfo(
        size_value=["user"], datasrc_class=LocalForecastingDataSource
    ),
    # --- 时空预测数据集 ---
    "large_st_forecast": DatasetInfo(
        size_value=["large", "small"],
        datasrc_class=LocalStForecastingDataSource,
    ),
    "small_st_forecast": DatasetInfo(
        size_value=["small"], datasrc_class=LocalStForecastingDataSource
    ),
    "user_st_forecast": DatasetInfo(
        size_value=["user"], datasrc_class=LocalStForecastingDataSource
    ),
    # --- 异常检测数据集（最重要，LaGraph主要用这个）---
    "large_detect": DatasetInfo(
        size_value=["large", "small"],
        datasrc_class=LocalAnomalyDetectDataSource,
    ),
    "small_detect": DatasetInfo(
        size_value=["small"], datasrc_class=LocalAnomalyDetectDataSource
    ),
    "user_detect": DatasetInfo(
        size_value=["user"], datasrc_class=LocalAnomalyDetectDataSource
    ),
}


def filter_data(
    metadata: pd.DataFrame, size_value: List[str], feature_dict: Optional[Dict] = None
) -> List[str]:
    """
    根据条件过滤数据集，返回符合条件的文件名列表
    
    工作原理：
    1. 首先按 feature_dict 条件过滤（如筛选特定场景的数据）
    2. 再按 size_value 条件过滤（如只取小规模数据集）
    3. 返回所有匹配的文件名
    
    参数：
        metadata:     元数据表（DataFrame），包含所有数据集的基本信息
        size_value:   允许的大小值列表，如 ["small"] 表示只取小数据集
        feature_dict: 额外的过滤条件字典，如 {"scenario": "machine_failure"}
    
    返回：符合条件的CSV文件名列表
    """
    # 去掉 feature_dict 中值为 None 的项（None 表示不限制该项）
    if feature_dict is not None:
        feature_dict = {k: v for k, v in feature_dict.items() if v is not None}

    # 用 reduce + and_ 组合多个条件（相当于多条件 AND 过滤）
    # 例如：同时满足 size=small 且 scenario=machine_failure
    filt_metadata = metadata
    if feature_dict:
        filt_metadata = metadata[
            reduce(and_, (metadata[k] == v for k, v in feature_dict.items()))
        ]
    # 再过滤 size 字段
    filt_metadata = filt_metadata[filt_metadata["size"].isin(size_value)]

    return filt_metadata["file_name"].tolist()


def _get_model_names(model_names: List[str]):
    """
    处理模型重名问题：给重名的模型加上后缀编号
    
    例如：["LaGraph", "LaGraph", "LaGraph"] 
    会被重命名为：["LaGraph", "LaGraph_1", "LaGraph_2"]
    
    为什么需要这个？
    - 同一个模型可能用不同参数跑多次
    - 重名会导致保存结果时文件互相覆盖
    
    参数：model_names 原始模型名列表
    返回：重命名后的模型名列表
    """
    s = pd.Series(model_names)
    cumulative_counts = s.groupby(s).cumcount()
    return [
        f"{model_name}_{cnt}" if cnt > 0 else model_name
        for model_name, cnt in zip(model_names, cumulative_counts)
    ]


def pipeline(
    data_config: dict,
    model_config: dict,
    evaluation_config: dict,
    save_path: str,
) -> List[str]:
    """
    基准测试主流水线：完整执行数据加载→模型构建→评估→报告
    
    这是整个框架的入口函数，协调所有模块完成一次完整的基准测试运行。
    
    参数：
        data_config:       数据配置（用哪个数据集、过滤条件等）
        model_config:      模型配置（用哪些模型、各自的参数等）
        evaluation_config: 评估配置（评估策略、指标、参数等）
        save_path:         结果保存路径（相对于 result 文件夹）
    
    返回：所有生成的日志文件名列表
    
    流程详解：
        1. 解析数据集名称，检查是否合法
        2. 确定数据源类型和需要加载的数据文件
        3. 启动数据服务器（异步数据供给）
        4. 根据模型配置创建所有模型实例
        5. 依次评估每个模型，收集结果
        6. 保存所有评估结果到CSV文件
    """
    # ============ 阶段1: 准备数据 ============
    # 获取数据集名称列表，默认用 small_forecast
    dataset_name_list = data_config.get("data_set_name", ["small_forecast"])
    if not dataset_name_list:
        dataset_name_list = ["small_forecast"]
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    
    # 检查所有数据集名称都在预定义注册表中
    for dataset_name in dataset_name_list:
        if dataset_name not in PREDEFINED_DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name}.")

    # 确保所有数据集使用相同类型的数据源（不能混用预测+异常检测）
    data_src_type = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class
    if not all(
        PREDEFINED_DATASETS[dataset_name].datasrc_class is data_src_type
        for dataset_name in dataset_name_list
    ):
        raise ValueError("Not supporting different types of data sources.")

    # 创建数据源实例
    data_src: DataSource = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class()
    
    # 确定要加载的具体数据文件
    data_name_list = data_config.get("data_name_list", None)
    if not data_name_list:
        # 没有指定具体文件，自动根据条件过滤
        data_name_list = []
        for dataset_name in dataset_name_list:
            size_value = PREDEFINED_DATASETS[dataset_name].size_value
            feature_dict = data_config.get("feature_dict", None)
            data_name_list.extend(
                filter_data(
                    data_src.dataset.metadata, size_value, feature_dict=feature_dict
                )
            )
    data_name_list = list(set(data_name_list))  # 去重
    if not data_name_list:
        raise ValueError("No dataset specified.")
    
    # 加载所有指定的数据文件
    data_src.load_series_list(data_name_list)
    # 启动全局存储数据服务器（异步进程，预先加载数据供后续使用）
    data_server = GlobalStorageDataServer(data_src, ParallelBackend())
    data_server.start_async()

    # ============ 阶段2: 模型构建 ============
    # 根据配置创建所有需要的模型实例
    # get_models 会解析配置中的模型列表，返回对应的模型工厂
    model_factory_list = get_models(model_config)

    # ============ 阶段3: 模型评估 ============
    # 每个模型在所有数据上评估一遍
    result_list = [
        eval_model(model_factory, data_name_list, evaluation_config)
        for model_factory in model_factory_list
    ]
    
    # 处理模型名称（处理重名问题）
    model_save_names = [
        it.split(".")[-1]
        for it in _get_model_names(
            [model_factory.model_name for model_factory in model_factory_list]
        )
    ]

    # ============ 阶段4: 保存结果 ============
    # 把每个模型的每份评估结果保存为CSV文件
    log_file_names = []
    for model_factory, result_itr, model_save_name in zip(
        model_factory_list, result_list, model_save_names
    ):
        for i, result_df in enumerate(result_itr.collect()):
            log_file_names.append(
                save_log(
                    result_df,
                    save_path,
                    model_save_name if i == 0 else f"{model_save_name}-{i}",
                )
            )

    return log_file_names
