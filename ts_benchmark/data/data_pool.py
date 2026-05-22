# -*- coding: utf-8 -*-
"""
数据池访问器（DataPool）
========================
全局数据池的单例包装器，提供统一的访问入口。

DataPool 本身不实现任何数据管理逻辑，它只是一个"代理"，
实际的数据管理在 DataPoolImpl 中完成。

为什么需要这一层？
  它利用单例模式（Singleton）确保整个框架中只有一个全局数据池实例。
  任何地方的代码都可以通过 DataPool().get_pool() 拿到同一个数据池，
  然后获取数据，避免了数据重复加载的问题。

设计模式：
  - Singleton（单例模式）: 全局唯一实例
  - Proxy（代理模式）: DataPool 只是 DataPoolImpl 的访问代理

典型用法：
  from ts_benchmark.data.data_pool import DataPool
  pool = DataPool().get_pool()  # 获取真正的数据池
  df = pool.get_data("MSL.csv")  # 获取数据
"""
from typing import NoReturn

from ts_benchmark.data.data_pool_impl_base import DataPoolImpl
from ts_benchmark.utils.design_pattern import Singleton


class DataPool(metaclass=Singleton):
    """
    全局数据池单例访问器

    它就像一个"前台接待员"：
    - 整个程序只有一位（单例）
    - 它不亲自干活（只是代理）
    - 它指引你去正确的"后台"（DataPoolImpl）

    职责：
    1. 持有 DataPoolImpl 实例的引用
    2. 提供 set_pool / get_pool 来切换/访问底层数据池
    """

    def __init__(self):
        # 真正的数据池对象，初始为空
        self.pool = None

    def set_pool(self, pool: DataPoolImpl) -> NoReturn:
        """
        设置底层的数据池实现

        通常在 Pipeline 初始化时调用，将配置好的 DataPoolImpl
        注入到全局 DataPool 单例中。

        参数：
            pool: 一个实现了 DataPoolImpl 接口的数据池对象
        """
        self.pool = pool

    def get_pool(self) -> DataPoolImpl:
        """
        获取底层的数据池实现

        框架中所有需要访问数据的地方都通过这个方法获取数据池。
        如果 pool 为 None，说明还没有初始化数据池（需要先调用 set_pool）。

        返回：
            当前底层数据池对象
        """
        return self.pool
