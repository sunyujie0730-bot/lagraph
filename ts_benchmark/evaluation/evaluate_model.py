# -*- coding: utf-8 -*-
"""
模型评估执行模块（Evaluate Model）
===================================
这是整个基准测试框架的"调度中心"，负责：
  1. 根据配置创建评估策略和评估器
  2. 将模型评估任务提交到并行后端执行
  3. 收集所有评估结果并打包成标准 DataFrame

整体工作流程：
  eval_model()                  ← 入口：创建策略 + 提交任务
    └→ ParallelBackend.schedule()  ← 并行调度（可能是 Ray 多进程 或 Sequential 单进程）
       └→ Strategy.execute()      ← 每个任务实际执行训练+推理+打分
          └→ EvalResult.collect()  ← 收集所有结果，打包成 DataFrame

类比理解：
  就像"高考阅卷系统"：
  - eval_model() 安排所有阅卷老师（并行后端）
  - Strategy.execute() 每个老师独立批改试卷（训练+推理+打分）
  - EvalResult.collect() 把所有老师的分数汇总成一个总成绩单

本模块关键组件：
  - _safe_execute(): 容错包装器（某个数据崩了不影响其他，像试卷丢了跳过不算）
  - eval_model():    主入口函数，创建策略、提交并行任务
  - EvalResult:      结果收集器，延迟收集 + 批量打包
  - build_result_df(): 将结果列表转为标准 DataFrame
"""
import functools
import json
import logging
import traceback
from typing import Callable, Tuple, List, Generator

import pandas as pd
import tqdm

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy import STRATEGY
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.parallel import ParallelBackend, TaskResult

logger = logging.getLogger(__name__)


def _safe_execute(fn: Callable, args: Tuple, get_default_result: Callable):
    """
    安全执行函数（容错包装器）

    作用：像"安全气囊"一样，即使被包装的函数崩溃了，
    也不会导致整个实验中断，而是返回默认结果。

    为什么需要它？
    在批量评估中，可能有几百个数据集 × 几十个模型组合。
    如果某个数据上某一步出错（如数据格式不对、模型不兼容），
    就中断整个流程，那前面的结果全白跑了。

    使用 _safe_execute 后：
    - 某个组合崩了 → 返回默认值（标记为 NaN + 错误日志）
    - 其他组合正常 → 继续执行，不受影响
    - 最终报告里能看到哪个组合失败了以及原因

    参数：
        fn:                要执行的函数
        args:              函数参数元组
        get_default_result: 失败时返回默认值的回调函数（通常是 strategy.get_default_result）

    返回：
        成功时返回 fn(*args) 的结果
        失败时返回 get_default_result() 的结果（含错误日志）
    """
    try:
        return fn(*args)
    except Exception as e:
        # 记录完整的错误堆栈和异常信息
        log = f"{traceback.format_exc()}\n{e}"
        return get_default_result(**{FieldNames.LOG_INFO: log})


class EvalResult:
    """
    评估结果收集器 —— "延迟收集" + "批量打包"模式

    这是并行评估的核心设计之一：任务的提交和结果的收集是分离的。

    为什么要分离？
    在并行模式下，所有模型可以在不同 Worker 上同时执行。
    主线程不需要等着每个完成——先全部提交（schedule），
    然后再统一收集（collect）。这样最大化并行效率。

    工作流程：
    1. eval_model() 提交所有任务 → 得到 TaskResult 列表
    2. 创建 EvalResult(result_list) → 包装结果列表
    3. 调用 collect() → 逐个等待完成 + 批量打包成 DataFrame

    两个关键优化：
    1. 懒收集：只在主动调用 collect() 时才开始等待结果
    2. 分批打包：当收集结果超过100万条时拆成多个 DataFrame
       （避免单次内存过大，方便写入文件）

    用法：
        eval_result = eval_model(model_factory, series_list, config)
        for df in eval_result.collect():
            df.to_csv("results.csv", mode='a')  # 分批写入文件
    """

    def __init__(
        self,
        strategy: Strategy,
        result_list: List[TaskResult],
        model_factory: ModelFactory,
        series_list: List[str],
    ):
        """
        初始化结果收集器

        参数：
            strategy:      评估策略实例（如 FixedDetectScore）
            result_list:   并行任务的结果句柄列表
                         （每个 TaskResult 可以通过 .result() 获取实际值）
            model_factory: 模型工厂（用于获取模型名和超参数信息）
            series_list:   所有数据序列的名称列表
                         （与 result_list 一一对应，用于在结果中标记数据来源）
        """
        self.result_list = result_list
        self.strategy = strategy
        self.model_factory = model_factory
        self.series_list = series_list

    def collect(self) -> Generator[pd.DataFrame, None, None]:
        """
        收集所有评估结果并打包成 DataFrame

        这是延迟收集的核心方法，返回一个生成器（Generator）。
        使用生成器的好处：
        - 不需要一次性把所有结果加载到内存
        - 分批返回 DataFrame，内存友好
        - 调用方能遍历处理每个批次

        分批逻辑：
        - 收集器累积到超过 100,000 条时触发一次打包
        - 每 100 个任务显示一次进度条（min_interval 控制刷新频率）
        - 最后不足批量的结果也打包返回

        返回：
            生成器，每次 yield 一个 pd.DataFrame
        """
        collector = self.strategy.get_collector()  # 获取策略的结果收集器

        # 动态调整进度条刷新频率：任务多的时候降低刷新频率（减少性能开销）
        min_interval = (
            0 if len(self.result_list) < 100 else 0.1
        )  # <100 个任务实时刷新，>=100 每 0.1 秒刷新

        for i, result in enumerate(
            tqdm.tqdm(
                self.result_list,
                desc=f"collecting {self.model_factory.model_name}",  # 进度条标题：显示模型名
                mininterval=min_interval,
            )
        ):
            # 安全获取结果：如果该任务执行失败，用默认结果替代
            # result.result 是 TaskResult 的方法，如果没有结果则阻塞等待
            collector.add(
                _safe_execute(
                    result.result,  # 等待并获取结果
                    (),             # 无参数
                    functools.partial(
                        self.strategy.get_default_result,  # 失败时的回退函数
                        **{FieldNames.FILE_NAME: self.series_list[i]},  # 预设文件名
                    ),
                )
            )

            # 当累积结果数超过 100,000 条时，打包为一个 DataFrame 返回
            if collector.get_size() > 100000:
                result_df = build_result_df(
                    collector.collect(), self.model_factory, self.strategy
                )
                yield result_df
                collector.reset()  # 清空收集器，准备下一批

        # 处理剩余不足批量的结果
        if collector.get_size() > 0:
            result_df = build_result_df(
                collector.collect(), self.model_factory, self.strategy
            )
            yield result_df


def eval_model(
    model_factory: ModelFactory, series_list: list, evaluation_config: dict
) -> EvalResult:
    """
    评估单个模型在所有数据上的表现 —— 框架主入口

    这是整个评估框架最核心的入口函数。它做的事情：
    1. 解析配置 → 创建评估策略 + 评估器
    2. 提交所有数据集的任务到并行后端
    3. 返回结果收集器（供后续 collect）

    参数：
        model_factory:      模型工厂对象，调用它创建模型实例
                           （如 LaGraph 的工厂函数）
        series_list:        要评估的数据序列名称列表
                           （如 ["MSL.csv", "swat.csv"]）
        evaluation_config:  评估配置字典，包含：
                           - strategy_args: 策略配置（策略名称 + 参数如 train_test_split）
                           - metrics:       指标配置（指标列表 或 "all" 表示全部）

    返回：
        EvalResult 对象，调用 .collect() 获取实际 DataFrame

    典型用法：
        result = eval_model(la_graph_factory, ["MSL.csv", "swat.csv"], config)
        for df in result.collect():
            print(df)  # 每10万条结果打印一次
    """
    # ===== 步骤1: 根据配置名获取策略类 =====
    # STRATEGY 是一个注册表（dict），key 是策略名，value 是策略类
    # 如 "fixed_detect_label" → FixedDetectLabel 类
    strategy_class = STRATEGY.get(evaluation_config["strategy_args"]["strategy_name"])
    if strategy_class is None:
        raise RuntimeError("strategy_class is none")

    # ===== 步骤2: 解析评估指标配置 =====
    # 支持三种配置格式：
    #   "all"              → 使用该策略支持的全部指标
    #   "f1"               → 单个指标名
    #   [{"name": "f1", "threshold": 0.5}] → 带参数的指标
    metric = evaluation_config["metrics"]
    if metric == "all":
        # "all" 模式：自动获取该策略支持的所有指标
        metric = list(strategy_class.accepted_metrics())
    elif isinstance(metric, (str, dict)):
        # 单个指标：包装成列表
        metric = [metric]

    # 统一格式：每个指标变成 {"name": "指标名", ...参数}
    # 如果是字符串就直接用，已经是字典的就保持不变
    metric = [
        {"name": metric_info} if isinstance(metric_info, str) else metric_info
        for metric_info in metric
    ]

    # ===== 步骤3: 检查指标合法性 =====
    # 每种策略只支持特定类型的指标
    # Score 策略 → 支持 AUC, VUS 等（连续分数评估指标）
    # Label 策略 → 支持 F1, Precision, Recall 等（标签评估指标）
    invalid_metrics = [
        m.get("name")
        for m in metric
        if m.get("name") not in strategy_class.accepted_metrics()
    ]
    if invalid_metrics:
        raise RuntimeError(
            "The evaluation index to be evaluated does not exist: {}".format(
                invalid_metrics
            )
        )

    # ===== 步骤4: 创建评估器 =====
    # 评估器负责实际计算指标（将策略要的指标列表传给它）
    evaluator = Evaluator(metric)

    # ===== 步骤5: 创建评估策略实例 =====
    # 策略定义了"训练→推理→打分"的完整流程
    strategy = strategy_class(evaluation_config["strategy_args"], evaluator)

    # ===== 步骤6: 提交所有任务到并行后端 =====
    # ParallelBackend 可能是 Ray（多进程并行）或 Sequential（单进程串行）
    # 取决于启动脚本中是否配置了 Ray
    eval_backend = ParallelBackend()
    result_list = []
    for series_name in tqdm.tqdm(
        series_list, desc=f"scheduling {model_factory.model_name}"
    ):
        # 每一个 (数据序列, 模型) 组合提交为一个独立任务
        # strategy.execute 是实际执行函数，框架会在 Worker 进程中调用它
        # TODO: 未来可以优化数据模型，减少并行模式下的通信开销
        result_list.append(
            eval_backend.schedule(strategy.execute, (series_name, model_factory))
        )

    # ===== 步骤7: 返回结果收集器 =====
    # 此时任务已经提交，但结果还没收集（懒收集模式）
    # 调用方需要执行 .collect() 来获取实际 DataFrame
    return EvalResult(strategy, result_list, model_factory, series_list)


def build_result_df(
    result_list: List, model_factory: ModelFactory, strategy: Strategy
) -> pd.DataFrame:
    """
    将评估结果列表转换为标准化的 DataFrame

    这是结果"装箱"函数，负责添加所有元信息列：
    - MODEL_NAME:      模型名称（如 "LaGraph"）
    - STRATEGY_ARGS:   策略参数（如 train_test_split=0.8）
    - MODEL_PARAMS:    模型超参数（JSON 格式）
    - 策略定义的指标列（F1, Precision, Recall...）
    - FILE_NAME:       数据文件名
    - FIT_TIME:        训练耗时
    - INFERENCE_TIME:  推理耗时
    - 等等

    参数：
        result_list:   原始结果列表（策略.execute 的返回值）
        model_factory: 模型工厂
        strategy:      评估策略

    返回：
        标准的评估结果 DataFrame，列顺序固定、格式统一

    列顺序说明：
        MODEL_NAME → STRATEGY_ARGS → MODEL_PARAMS → [指标1, 指标2, ...] → 元信息列
        前3列是标识信息（谁、什么配置、什么参数），后面是实际分数和耗时
    """
    # 将原始结果列表转为 DataFrame，列名由策略定义
    result_df = pd.DataFrame(result_list, columns=strategy.field_names)

    # 插入模型超参数字段（如果不存在）
    # 某些模型支持超参数搜索，每个数据可能返回不同的参数
    # 如果已经存在就保留，否则用工厂的统一参数填充
    if FieldNames.MODEL_PARAMS not in result_df.columns:
        result_df.insert(
            0,
            FieldNames.MODEL_PARAMS,
            json.dumps(model_factory.model_hyper_params, sort_keys=True),
        )
    # 插入策略参数（如 train_test_split=0.8, threshold=0.5 等）
    result_df.insert(0, FieldNames.STRATEGY_ARGS, strategy.get_config_str())
    # 插入模型名称（最重要的一列）
    result_df.insert(0, FieldNames.MODEL_NAME, model_factory.model_name)

    # 检查是否有必需字段缺失（防御性检查）
    missing_fields = set(FieldNames.all_fields()) - set(result_df.columns)
    if missing_fields:
        raise ValueError(
            "These required fields are missing in the result df: {}".format(
                missing_fields
            )
        )
    return result_df
