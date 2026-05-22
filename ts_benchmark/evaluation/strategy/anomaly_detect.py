# -*- coding: utf-8 -*-
"""
异常检测评估策略模块（Anomaly Detect Strategy）
==============================================
定义了几种不同的异常检测评估策略，每种策略在"数据切分"和"结果类型"上有区别。

核心概念区分：
  1. Fixed vs UnFixed：训练/测试的切分方式不同
     - Fixed：按固定比例切分（如80%训练+20%测试）
     - UnFixed：用数据元信息中预设的训练长度切分
     - All：全量数据训练+全量测试（无监督场景）

  2. Score vs Label：模型输出的类型不同
     - Score：模型输出异常分数（float），需要给定阈值来判断
     - Label：模型直接输出异常标签（0/1）

类层次结构：
  Strategy (基类)
    └── AnomalyDetect (抽象类，定义通用流程)
          ├── FixedDetectScore    — 固定比例 + 输出分数
          ├── FixedDetectLabel    — 固定比例 + 输出标签
          ├── UnFixedDetectScore  — 预设长度 + 输出分数
          ├── UnFixedDetectLabel  — 预设长度 + 输出标签
          ├── AllDetectScore      — 全量数据 + 输出分数
          └── AllDetectLabel      — 全量数据 + 输出标签

类比理解：
  就像不同的"考试方案"：
  - Fixed模式：平时练习（用80%数据训练，20%验证效果）
  - UnFixed模式：用老师指定的复习范围
  - All模式：用整本书的内容练习
  - Score模式：只算概率/分数，需要确定多少分算及格
  - Label模式：直接给对/错答案
"""
import base64
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed, fix_all_random_seed


class AnomalyDetect(Strategy):
    """
    异常检测策略抽象基类
    
    定义了异常检测的通用评估流程：
    1. 切分数据（训练集/测试集）
    2. 训练模型（detect_fit 或 fit）
    3. 推理预测（detect）
    4. 评估指标（evaluator.evaluate）
    5. 记录结果并序列化保存模型
    
    这是模板方法模式：子类只需实现 split_data() 和 detect() 即可
    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化策略
        
        参数：
            strategy_config: 策略配置字典（如切分比例、阈值等）
            evaluator:       评估器实例，包含所有要计算的指标
        """
        super().__init__(strategy_config, evaluator)
        self.model = None       # 当前正在评估的模型实例
        self.data_lens = None   # 当前数据的总长度

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行完整的异常检测评估流程（核心入口）
        
        这是策略的主函数，被 pipeline 调用来评估单个模型在单个数据集上的表现
        
        参数：
            series_name:    数据序列名称（如 "MSL.csv"）
            model_factory:  模型工厂函数（调用它创建模型实例）
        
        返回：single_series_results_list 评估结果列表
        
        详细流程：
        1. 固定随机种子（保证可复现性）
        2. 创建模型实例
        3. 切分训练/测试数据
        4. 训练模型并计时
        5. 推理预测并计时
        6. 计算所有评估指标
        7. 序列化保存预测结果（备用）
        8. 异常处理：如果某步骤失败，返回默认结果（全NaN+错误日志）
        """
        fix_random_seed()  # 固定随机种子，确保实验可复现

        model = model_factory()
        try:
            self.model = model

            # ★ 打印当前数据集名称
            print(f"\n{'='*60}")
            print(f"  Dataset: {series_name}")
            print(f"{'='*60}")
            
            # ===== 步骤1: 切分数据 =====
            # 子类实现不同的切分策略（Fixed/UnFixed/All）
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            
            # ===== 步骤2: 训练模型 =====
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                # 优先用 detect_fit（异常检测专用训练接口）
                self.model.detect_fit(train_data, train_label)
            else:
                # 回退到通用 fit 接口
                self.model.fit(train_data, train_label)
            end_fit_time = time.time()
            
            # ===== 步骤3: 推理预测 =====
            predict_labels, another = self.detect(test_data)
            # 确保预测结果是字典格式（支持多异常率预测）
            if not isinstance(predict_labels, dict):
                predict_labels = {"None": predict_labels}

            actual_label = test_label.to_numpy().flatten()
            end_inference_time = time.time()

            # ===== 步骤4: 评估每个异常率的预测结果 =====
            single_series_results_list = []
            for ratio, predict_label in predict_labels.items():
                remaining_length = len(actual_label) - len(predict_label)
                print(remaining_length)
                
                # 如果预测序列比真实标签短，用0填充（保守处理：缺的地方都当没异常）
                if remaining_length > 0:
                    predict_label = np.pad(
                        predict_label,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )
                    another = np.pad(
                        another,
                        (0, remaining_length),
                        mode="constant",
                        constant_values=0,
                    )

                # 计算指标（带异常捕获）
                single_series_results, log_info = self.evaluator.evaluate_with_log(
                    actual=actual_label.astype(float),
                    predicted=predict_label.astype(float)
                )
                print(single_series_results)

                # ===== 步骤5: 序列化保存预测数据（用于事后分析） =====
                # 用 base64 编码 pickle 序列化结果，方便存在CSV中
                inference_data = [predict_label, another]
                
                actual_data_pickle = pickle.dumps(test_label)
                actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

                inference_data_pickle = pickle.dumps(inference_data)
                inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                    "utf-8"
                )
                
                # 拼接所有结果字段
                single_series_results += [
                    series_name,                       # 数据集名称
                    end_fit_time - start_fit_time,      # 训练耗时
                    end_inference_time - end_fit_time,  # 推理耗时
                    ratio,                              # 异常率
                    '',                                 # 原始数据占位（预留字段）
                    '',                                 # 推理数据占位（预留字段）
                    log_info,                           # 错误日志
                ]
                single_series_results_list.append(single_series_results)
        except Exception as e:
            # ===== 异常处理：评估失败时返回默认结果 =====
            log = f"The error series is: {series_name}\n{traceback.format_exc()}\n{e}"
            single_series_results_list = [self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )]
        return single_series_results_list

    def split_data(self, data: str):
        """切分训练/测试数据（子类实现）"""
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        """用模型进行异常检测推理（子类实现）"""
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        """声明该策略支持的评估指标类型（子类实现）"""
        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        """
        定义结果 DataFrame 的列名顺序
        
        结构：[指标1, 指标2, ..., 文件名, 训练时间, 推理时间, 异常率, 原始数据, 推理数据, 日志]
        """
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ANOMALY_RATIO,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]


# ============================================================
# 以下是6种具体的评估策略子类
# ============================================================

class FixedDetectScore(AnomalyDetect):
    """
    策略1：固定比例切分 + 输出异常分数（Score）
    
    适用场景：你需要模型输出异常分数（连续值），
            然后自己调阈值来决定多少分以上算异常
    
    切分方式：按 train_test_split 配置的比例切分
            如 0.8 表示前80%训练，后20%测试
    
    输出类型：detect_score() → 异常分数数组 [0.1, 0.9, 0.3, ...]
    """
    REQUIRED_FIELDS = ["train_test_split"]  # 配置中必须提供切分比例

    def split_data(self, series_name):
        """前 train_test_split 比例的数据用于训练，余下用于测试"""
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        # 分离特征和标签（label列是最后一列）
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        """调用模型的 detect_score 方法获取异常分数"""
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    """
    策略2：固定比例切分 + 输出异常标签（Label）
    
    与 FixedDetectScore 的区别：
    - Score 输出分数，需要额外选阈值
    - Label 直接输出 0/1 标签，更简单直接
    
    适用场景：模型天然输出二分类标签（正常=0，异常=1）
    """
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        """切分逻辑同 FixedDetectScore"""
        data = DataPool().get_pool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(self.strategy_config["train_test_split"] * self.data_lens)
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        """调用模型的 detect_label 方法直接获取标签"""
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    """
    策略3：数据集预设训练长度 + 输出异常分数（Score）
    
    与 Fixed 的区别：
    - Fixed 用固定比例（如80%）
    - UnFixed 用数据集元信息中的 train_lens 字段决定切分点
    
    适用场景：不同数据集有不同的预设训练/测试切分方案
    """
    def split_data(self, series_name: str):
        """从元信息读取预设的训练长度进行切分"""
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        # 从元信息读取 train_lens
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    """
    策略4：数据集预设训练长度 + 输出异常标签（Label）
    
    同 UnFixedDetectScore 但输出标签而非分数
    """
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        data = data.reset_index(drop=True)
        train_length = int(
            DataPool().get_pool().get_series_meta_info(series_name)["train_lens"].item()
        )
        train, test = split_before(data, train_length)
        train_data, train_label = (
            train.loc[:, train.columns != "label"],
            train.loc[:, ["label"]],
        )
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    """
    策略5：全量数据模式 + 输出异常分数（Score）
    
    训练和测试都用全部数据（无监督/自监督学习场景）
    
    特点：
    - train_label 为 None（不使用标签训练）
    - 训练数据 = 测试数据（全量）
    
    适用场景：无监督异常检测，模型从全量数据学习正常模式
    """
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data  # 全量做训练
        test = data   # 全量做测试
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    """
    策略6：全量数据模式 + 输出异常标签（Label）
    
    同 AllDetectScore 但输出标签
    """
    def split_data(self, series_name):
        data = DataPool().get_pool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != "label"], None
        test_data, test_label = (
            test.loc[:, train.columns != "label"],
            test.loc[:, ["label"]],
        )
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
