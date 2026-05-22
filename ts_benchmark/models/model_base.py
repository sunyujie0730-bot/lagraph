# -*- coding: utf-8 -*-
"""
模型基类模块（Model Base）
=========================
定义了所有基准测试兼容模型必须遵守的标准接口。

这是框架的"模型协议"：任何模型只要继承 ModelBase 并实现对应方法，
就能被框架自动加载、训练、评估，无需修改框架代码。

接口设计理念：
  采用"鸭子类型"——不关心模型内部实现，只要求暴露标准方法。
  只要你的模型有 detect_fit() + detect_score() 方法，
  框架就能在异常检测策略中调用它。

提供的接口方法：
  预测任务：
    - forecast_fit():  训练预测模型
    - forecast():      进行预测推理
    - batch_forecast(): 批量预测（可选）

  异常检测任务：
    - detect_fit():    训练异常检测模型（★最常用）
    - detect_score():  输出异常分数（连续值）
    - detect_label():  输出异常标签（0/1）

  通用方法：
    - fit():           通用训练接口（回退方案）
"""

import abc
from typing import Union, Optional, Dict

import numpy as np
import pandas as pd


def annotate(**kwargs):
    """
    函数装饰器：给函数添加或更新类型注解（annotations）
    
    用途：标记某些方法的实现状态（如 not_implemented_batch=True）
         框架在运行时可以通过 __annotations__ 检查该方法是否有默认实现
    
    参数：kwargs 要添加/更新的注解键值对
    
    返回值：装饰器函数
    """

    def wrapper(func):
        func.__annotations__.update(kwargs)
        return func

    return wrapper


class BatchMaker(metaclass=abc.ABCMeta):
    """
    批量数据生成器接口（预测任务用）
    
    职责：为大批量预测任务提供按批次组织的数据
    
    必须实现的方法：
        make_batch(batch_size, win_size) → 生成一批预测数据
    
    参数说明：
        batch_size: 每批包含的样本数
        win_size:   每次预测使用的窗口长度
    
    使用场景：当数据量太大无法一次性预测时，拆分成小批次处理
    """

    @abc.abstractmethod
    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        生成一批预测数据
        
        参数：
            batch_size: 批大小（本批包含多少个样本）
            win_size:   预测窗口长度
        
        返回：包含数据的字典
        """


class ModelBase(metaclass=abc.ABCMeta):
    """
    模型基类 —— 所有基准测试模型的"标准接口"
    
    这是框架最重要的抽象类。任何你想在框架中评估的模型，
    都应该继承 ModelBase 并实现对应方法。
    
    设计模式：模板方法 + 策略模式
    - 框架定义了"怎样评估"（策略）
    - 模型定义了"怎样预测"（具体实现）
    - 两者通过 ModelBase 接口解耦
    
    使用建议：
    - 做异常检测 → 实现 detect_fit() + detect_score() 或 detect_label()
    - 做预测任务 → 实现 forecast_fit() + forecast()
    """

    @abc.abstractmethod
    def forecast_fit(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        *,
        covariates: Optional[Dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs
    ) -> "ModelBase":
        """
        训练预测模型（预测任务专用）
        
        参数：
            train_data:        训练时序数据，形状 (时间步, 特征数)
            covariates:        协变量（辅助信息），如天气、节假日等
            train_ratio_in_tv: 训练/验证集分割比例
                               =1.0 表示不划分验证集
                               <1.0 表示划分验证集
        
        返回：训练好的模型对象（self）
        
        注意：正常数据通常不包含 label 列
        """

    @abc.abstractmethod
    def forecast(
        self,
        horizon: int,
        series: Union[pd.DataFrame, np.ndarray],
        covariates: Optional[Dict] = None,
        **kwargs
    ) -> np.ndarray:
        """
        用训练好的模型进行预测（预测任务专用）
        
        参数：
            horizon:    预测步长（预测未来多少个时间步）
            series:     输入的时间序列数据
            covariates: 协变量
        
        返回：预测结果的 numpy 数组
        """

    @annotate(not_implemented_batch=True)
    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        covariates: Optional[Dict] = None,
        **kwargs
    ) -> np.ndarray:
        """
        批量预测（可选实现，默认不支持）
        
        与 forecast() 的区别：
        - forecast(): 一次性预测所有数据
        - batch_forecast(): 分批次预测，适合数据量极大的场景
        
        如果模型没有实现此方法，框架会回退到逐条预测
        
        参数：
            horizon:     每次预测的步长
            batch_maker: 批次数据生成器
            covariates:  协变量
        
        返回：预测结果数组
        """
        raise NotImplementedError("Not implemented batch forecasting!")
