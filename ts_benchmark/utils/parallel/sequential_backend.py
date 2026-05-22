# -*- coding: utf-8 -*-
"""
顺序执行后端（串行模式）

这是并行框架在"单进程模式"下的实现。

与 RayBackend（真正并行）不同，SequentialBackend 所有任务
都在当前进程内顺序执行，不会有任何并行。

为什么需要串行后端？
  1. 调试友好：串行执行时日志不交错，堆栈清晰
  2. 环境简单：不需要安装 Ray，不依赖额外服务
  3. 资源受限：单 GPU 或小内存环境下更稳定
  4. 测试：确保单进程下的逻辑正确，再切换到 Ray 并行

全部知识点：
  - SequentialResult：立即存储结果的 TaskResult（无阻塞）
  - SequentialSharedStorage：基于普通 dict 的共享存储
  - SequentialBackend：串行后的总入口，统一接口

类比：SequentialBackend 就像一个只有一个厨师的后厨，
客人点一个菜做一道，RayBackend 则是五个厨师同时开工。
"""

from __future__ import absolute_import

import os
import warnings
from typing import Tuple, Any, NoReturn, Callable, Optional, List, Dict

from ts_benchmark.utils.parallel.base import TaskResult, SharedStorage


class SequentialResult(TaskResult):
    """
    立即就绪的结果容器

    与 RayResult 不同（Ray 版需要等任务完成），
    SequentialResult 在 put() 时结果就已完成。
    调用 result() 直接返回，不会阻塞。

    实现方式：直接用 Python 变量存储结果。
    """

    def __init__(self):
        self._result = None  # 任务结果存储

    def result(self) -> Any:
        """立刻返回结果（因为串行下已计算完毕）"""
        return self._result

    def put(self, value: Any) -> NoReturn:
        """存储计算后的结果"""
        self._result = value


class SequentialSharedStorage(SharedStorage):
    """
    基于普通 dict 的共享存储

    串行模式下共享存储就是 Python 字典。
    因为只有一个进程，不需要多进程同步机制。

    用法：
        storage = SequentialSharedStorage()
        storage.put("model", my_model)
        model = storage.get("model")
    """

    def __init__(self):
        self.storage = {}  # 普通的 Python 字典

    def put(self, name: str, value: Any) -> NoReturn:
        """将变量存入字典"""
        self.storage[name] = value

    def get(self, name: str, default_value: Any = None) -> Any:
        """从字典中取变量，不存在则返回默认值"""
        return self.storage.get(name, default_value)


class SequentialBackend:
    """
    顺序执行后端 —— 串行任务的统一入口

    这是并行框架在单进程下的实现。所有任务在当前进程
    内 / 一个接一个 / 顺序执行。

    主要功能：
    - 管理共享存储
    - 管理 CUDA 设备可见性
    - 调度任务（但只是同步执行）
    - 提供与 RayBackend 完全相同的接口

    用法：
        backend = SequentialBackend(gpu_devices=[0])
        backend.init()
        result = backend.schedule(my_func, (arg1, arg2))
        value = result.result()
        backend.close()
    """

    def __init__(self, gpu_devices: Optional[List[int]] = None, **kwargs):
        super().__init__()
        # 可用的 GPU 设备列表，如 [0, 1] 表示使用 GPU 0 和 1
        self.gpu_devices = gpu_devices if gpu_devices is not None else []
        self.storage = None  # 初始化后变成 SequentialSharedStorage

    def init(self) -> NoReturn:
        """初始化后端：创建共享存储 + 设置 CUDA 设备"""
        self.storage = SequentialSharedStorage()
        # 设置 CUDA_VISIBLE_DEVICES 环境变量
        # 例如 gpu_devices=[0, 2] → "0,2"
        # PyTorch 只对编号中的 GPU 可见
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_devices))

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> SequentialResult:
        """
        调度一个任务（串行模式下直接执行）

        参数：
            fn: 要执行的函数
            args: 函数的参数元组
            timeout: 超时时间（串行模式不支持，忽略）

        返回：
            SequentialResult: 包含执行结果的对象

        注意：timeout 在串行模式下无效，直接用同步方式执行。
        """
        if timeout != -1:
            warnings.warn("timeout is not supported by SequentialBackend, ignoring")
        res = SequentialResult()
        # 直接同步执行：调用函数，结果立刻存储
        res.put(fn(*args))
        return res

    def close(self, force: bool = False):
        """关闭后端（串行模式无需释放资源）"""
        pass

    @property
    def shared_storage(self) -> SharedStorage:
        """获取共享存储实例"""
        return self.storage

    @property
    def env(self) -> Dict:
        """获取环境信息（供任务使用）"""
        return {
            "storage": self.shared_storage,
        }

    def execute_on_workers(self, func: Callable) -> NoReturn:
        """
        在所有 worker 上执行函数（串行模式下只在当前进程执行）
        """
        func(self.env)

    def add_worker_initializer(self, func: Callable) -> NoReturn:
        """添加 worker 的初始化函数（串行模式忽略）"""
        pass
