# -*- coding: utf-8 -*-
"""
模型加载器（Model Loader）
==========================
负责动态加载、解析和实例化各类异常检测模型。

这是整个框架的"模型工厂"层，它解决了以下问题：
  1. 用户只需在配置文件中写模型路径（字符串），不用手动 import
  2. 自动解析模型信息（工厂函数 + 超参数 + 适配器）
  3. 支持统一配置推荐超参数，方便公平对比不同模型

核心流程：
  配置文件 model_name 字符串
    → import 对应模块（动态导入）
    → 解析模型信息（dict 或 callable）
    → 合并超参数（推荐参数 + 用户参数）
    → 创建 ModelFactory → 调用后得到模型实例

类比理解：
  就像餐厅点菜系统：
  - model_name 是菜名（如 "self_impl.LaGraph.LaGraph.LaGraph"）
  - model_loader 是服务员，负责找到这道菜并交给后厨
  - ModelFactory 是后厨的流水线，按菜单（超参数）做出菜（模型实例）
"""
import importlib
import logging
from typing import Any, Union, Dict, Callable, List

from ts_benchmark.baselines import ADAPTER

logger = logging.getLogger(__name__)


def _import_attribute(attr_path: str) -> Any:
    """
    根据完整的属性路径动态导入属性

    这是动态加载的核心函数。给定一个完整的模块路径，
    它会导入对应的 Python 属性（类、函数、字典等）。

    支持字符串重定向（用于短名称懒加载）：
    如果解析结果是一个字符串，会递归地将它作为新路径重新解析。

    例如：attr_path = "ts_benchmark.baselines.LaGraph"
    → 解析到字符串 "ts_benchmark.baselines.self_impl.LaGraph.LaGraph.LaGraph"
    → 自动重定向到完整路径并加载 LaGraph 类

    参数：
        attr_path: 用点号分隔的完全限定路径
    返回：
        如果目标属性存在则返回该属性，否则返回 None
    """
    # 从右往左分割：最后一个点之前是包路径，之后是属性名
    package_name, name = attr_path.rsplit(".", 1)

    # 动态导入模块（等同于 importlib.import_module）
    package = importlib.import_module(package_name)
    # 从模块中取出属性，找不到就返回 None
    result = getattr(package, name, None)
    # 字符串重定向：支持短名称懒加载到完整路径
    if isinstance(result, str):
        return _import_attribute(result)
    return result


def import_model_info(model_path: str) -> Union[Dict, Callable]:
    """
    导入模型信息

    模型信息有两种格式：

    格式一：字典（推荐）
    {
        "model_factory": LaGraph,          # ★必填：模型的工厂函数（类或函数）
        "model_hyper_params": {},          # 可选：模型的默认超参数
        "required_hyper_params": {},       # 可选：需要框架提供的超参数
        "model_name": "LaGraph"            # 可选：显示在日志中的名字
    }

    格式二：可调用对象
    直接返回一个类或工厂函数作为 model_info

    关于 required_hyper_params（关键机制）：
      某些模型的超参数可能依赖数据集的信息来决定，例如窗口大小。
      如果每个模型自己写死窗口大小，对比就不公平了。
      有了 required_hyper_params，模型可以把这些参数"外包"给框架：
      - 模型声明 {"win_size": "recommend_win_size"}
      - 框架统一设 recommend_win_size = 100
      - 所有模型都用一样的窗口大小 → 公平对比！

    参数：
        model_path: 模型信息的完全限定路径（如 "self_impl.LaGraph.LaGraph.MODEL"）
    返回：
        导入的模型信息（字典或可调用对象）
    抛出：
        ValueError: 如果模型信息类型不符合要求
    """
    model_info = _import_attribute(model_path)

    if not isinstance(model_info, (Dict, Callable)):
        raise ValueError(
            f"Unsupported model info with type {type(model_info).__name__}"
        )

    return model_info


def get_model_info(model_config: Dict) -> Union[Dict, Callable]:
    """
    根据模型配置获取模型信息

    模型路径的搜索策略（按优先级从高到低）：
    1. 如果 model_name 以 "global." 开头，去掉前缀后作为路径
    2. "ts_benchmark.baselines." + model_name
    3. model_name 本身（完全限定路径）

    这样设计的好处：
    - 短路径（如 "self_impl.LaGraph.LaGraph.LaGraph"）会自动补全
    - 完全限定路径也能正常工作
    - "global." 前缀支持引用全局注册的模型

    还支持适配器（adapter）机制：
    - 适配器可以对模型信息进行包装/修改
    - 例如：time_series_library 适配器可能调整输入输出格式

    参数：
        model_config: 模型配置字典，支持的字段：
            - model_name (str): 模型路径
            - adapter (str, 可选): 适配器名称
    返回：
        模型信息（字典或可调用对象）
    抛出：
        ImportError 或 AttributeError: 所有搜索路径都找不到模型时
    """
    # 构建搜索路径列表
    model_name_candidates = [
        model_config["model_name"][7:] if model_config["model_name"].startswith("global.") else None,
        "ts_benchmark.baselines." + model_config["model_name"],
        model_config["model_name"],
    ]
    # 过滤掉 None 值
    model_name_candidates = list(filter(None, model_name_candidates))

    model_info = None
    for model_name in model_name_candidates:
        try:
            logger.info("Trying to load model %s", model_name)
            model_info = import_model_info(model_name)
        except (ImportError, AttributeError, ValueError):
            logger.info("Loading model %s failed", model_name)
            continue
        else:
            break  # 找到了就停止搜索

    # 如果配置了适配器，对模型信息进行包装
    adapter_name = model_config.get("adapter")
    if adapter_name is not None:
        if adapter_name not in ADAPTER:
            raise ValueError(f"Unknown adapter {adapter_name}")
        # 适配器是一个函数，接收原始 model_info，返回包装后的 model_info
        model_info = _import_attribute(ADAPTER[adapter_name])(model_info)

    return model_info


def get_model_hyper_params(
    recommend_model_hyper_params: Dict, required_hyper_params: Dict, model_config: Dict
) -> Dict:
    """
    获取模型最终使用的超参数

    超参数的合并策略：
    1. 先用框架推荐的超参数填充（如果模型有 required_hyper_params 声明）
    2. 再用配置文件中用户指定的超参数覆盖（用户优先）

    举个例子：
        框架推荐:  {"recommend_win_size": 100}
        模型声明:  {"win_size": "recommend_win_size"}
        用户配置:  {"model_hyper_params": {"win_size": 200}}

        最终结果:  {"win_size": 200}
        （框架推荐 100 → 用户覆盖为 200）

    参数：
        recommend_model_hyper_params: 框架推荐的超参数字典
        required_hyper_params: 模型要求框架提供的超参数映射
            {模型参数名: 框架标准参数名}
        model_config: 模型配置

    返回：
        合并后的超参数字典
    抛出：
        ValueError: 如果有必需的超参数缺失
    """
    # 第一步：映射框架推荐参数到模型参数
    model_hyper_params = {
        arg_name: recommend_model_hyper_params[arg_std_name]
        for arg_name, arg_std_name in required_hyper_params.items()
        if arg_std_name in recommend_model_hyper_params
    }
    # 第二步：用户配置覆盖
    model_hyper_params.update(model_config.get("model_hyper_params", {}))

    # 检查是否还有未填充的必需参数
    missing_hp = set(required_hyper_params) - set(model_hyper_params)
    if missing_hp:
        raise ValueError("These hyper parameters are missing : {}".format(missing_hp))
    return model_hyper_params


class ModelFactory:
    """
    模型工厂：标准化的模型实例化器

    它就像一个"流水线"，记住：
    - 要生产什么产品（model_factory 是什么类/函数）
    - 用什么配方（model_hyper_params 超参数）
    - 产品名字是什么（model_name）

    调用它（__call__）就能得到一个训练好的模型实例。

    典型用法：
        factory = ModelFactory("LaGraph", LaGraph, {"win_size": 100})
        model = factory()  # 等同于 LaGraph(win_size=100)
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable,
        model_hyper_params: dict,
    ):
        """
        初始化模型工厂

        参数：
            model_name: 模型名称（用于日志记录）
            model_factory: 模型工厂函数（类或普通函数）
            model_hyper_params: 创建模型时传入的超参数字典
        """
        self.model_name = model_name
        self.model_factory = model_factory
        self.model_hyper_params = model_hyper_params

    def __call__(self) -> Any:
        """
        实例化模型

        调用 model_factory(**model_hyper_params)，等价于：
            LaGraph(win_size=100, lr=0.001, ...)

        返回：
            一个兼容 ModelBase 接口的模型实例
        """
        return self.model_factory(**self.model_hyper_params)


def get_models(all_model_config: Dict) -> List[ModelFactory]:
    """
    根据配置批量创建模型工厂列表

    这是模型加载的顶层入口函数。
    例如配置文件中写了对 3 个模型做测试，这个函数就会返回 3 个 ModelFactory。

    流程图：
    all_model_config["models"]
      → 遍历每个模型配置
        → get_model_info → 获取模型信息
        → get_model_hyper_params → 合并超参数
        → ModelFactory → 创建工厂
      → 返回工厂列表

    参数：
        all_model_config: 完整的模型配置字典，支持的字段：
            - models (list): 模型配置列表，每个元素是一个模型配置字典
            - recommend_model_hyper_params (dict, 可选): 全局推荐超参数

    返回：
        ModelFactory 对象列表，用于后续逐个实例化模型
    """
    model_factory_list = []

    for model_config in all_model_config["models"]:
        # 第一步：获取模型信息
        model_info = get_model_info(model_config)
        # 备用名称：取 model_name 的最后一段（如 "LaGraph"）
        fallback_model_name = model_config["model_name"].split(".")[-1]

        # 第二步：解析模型信息
        if isinstance(model_info, Dict):
            # 字典格式：提取工厂、超参数声明、名称
            model_factory = model_info.get("model_factory")
            if model_factory is None:
                raise ValueError("model_factory is none")
            required_hyper_params = model_info.get("required_hyper_params", {})
            model_name = model_info.get("model_name", fallback_model_name)
        elif isinstance(model_info, Callable):
            # 可调用对象格式：直接作为工厂函数
            model_factory = model_info
            required_hyper_params = {}
            if hasattr(model_factory, "required_hyper_params"):
                required_hyper_params = model_factory.required_hyper_params()
            model_name = fallback_model_name
        else:
            raise ValueError(f"Unexpected model info type {type(model_info).__name__}")

        # 第三步：合并超参数
        model_hyper_params = get_model_hyper_params(
            all_model_config.get("recommend_model_hyper_params", {}),
            required_hyper_params,
            model_config,
        )

        # 第四步：创建模型工厂并加入列表
        model_factory_list.append(
            ModelFactory(model_name, model_factory, model_hyper_params)
        )

    return model_factory_list
