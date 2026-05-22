# -*- coding: utf-8 -*-
"""
并行计算基础设施 —— 抽象基类

定义并行任务系统中两个核心抽象：

1. TaskResult —— 异步任务的结果句柄
   - 像"外卖单号"：提交任务后得到一个 TaskResult 对象，
     之后随时可以查看任务是否完成、获取结果。
   - 如果结果还没算完，调用 .result() 会阻塞等待。

2. SharedStorage —— 跨任务共享的存储
   - 像"共享黑板"：多个任务之间可以读写同一个变量。
   - 每个任务都能 put（写入）和 get（读取），
     信息在不同的并行任务之间流动。

这两个接口定义了"合同"，具体实现由子类完成：
  - Sequential*：单进程顺序执行（无真正并行）
  - Ray*：基于 Ray 的真并行实现
"""

from __future__ import absolute_import

import abc
from typing import Any, NoReturn


class TaskResult(metaclass=abc.ABCMeta):
    """
    任务结果句柄（抽象基类）

    框架提交一个任务后，立刻返回一个 TaskResult 对象。
    调用 result() 方法会阻塞，直到任务完成并返回结果。

    类比：你点了外卖，拿到一个订单号（TaskResult）。
    你可以随时调用 .result() 来"取餐"，如果还没做好就等着。

    子类需要实现：
    - result(): 阻塞等待并返回任务结果
    - put(value): 写入结果值（由执行器调用）
    """

    @abc.abstractmethod
    def result(self) -> Any:
        """
        阻塞等待，直到任务完成，然后返回结果
        """

    @abc.abstractmethod
    def put(self, value: Any) -> NoReturn:
        """
        设置结果值（由任务执行器调用）
        """


class SharedStorage(metaclass=abc.ABCMeta):
    """
    跨任务共享存储（抽象基类）

    允许多个并行任务共享数据。任何一个任务可以：
    - put(name, value)：将 value 以 name 为键存储
    - get(name)：按 name 取出之前存储的值

    类比：一个共享白板，任何任务都可以在上面写东西（put），
    也可以看别人写了什么（get）。

    子类需要实现：
    - put(name, value): 存储变量
    - get(name, default_value): 获取变量（不存在则返默认值）
    """

    @abc.abstractmethod
    def put(self, name: str, value: Any) -> NoReturn:
        """
        将一个变量存储到共享存储中
        """

    @abc.abstractmethod
    def get(self, name: str, default_value: Any = None) -> Any:
        """
        从共享存储中读取变量
        如果不存在，返回 default_value
        """
