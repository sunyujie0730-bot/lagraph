# -*- coding: utf-8 -*-
"""
Ray 并行后端（真·多进程并行）

这是并行框架在 Ray 分布式系统上的实现，是项目的核心并行引擎。

Ray 是什么？
  Ray 是一个分布式计算框架，让 Python 代码可以轻松地在多个
  CPU/GPU 和多台机器上并行执行。就像"Python 版的 MPI"，
  但使用起来简单很多。

为什么用 Ray 做后端？
  1. 真正的多进程并行：每个 Ray Actor 一个独立进程
  2. GPU 资源隔离：每个 Actor 分配特定 GPU，不冲突
  3. 超时机制：每个任务可以限时，超时就杀掉重启
  4. 自动重启：Actor 崩溃后自动复活（max_restarts=-1）
  5. 共享存储：通过 Ray 的 ObjectRef 实现跨进程数据共享

本模块包含以下组件（从底层到顶层）：

  RayActor —— 单个工作进程的抽象
    ↓
  RayActorPool —— 管理一组 Actor 的池子（调度/超时/重启）
    ↓
  RayBackend —— 顶层入口，统一的 schedule/close 接口
    ↓
  RaySharedStorage —— 基于 Ray ObjectRef 的跨进程共享存储
    ↓
  RayResult —— 异步任务的结果句柄（用 Event 实现阻塞等待）

整体架构类比：
  RayActor = 厨师（一个人）
  RayActorPool = 后厨（管理五个厨师，分配订单）
  RayBackend = 餐厅前台（接单 → 交给后厨 → 返回取餐号）
"""

from __future__ import absolute_import

import itertools
import logging
import os
import queue
import sys
import threading
import time
from typing import Callable, Tuple, Any, List, NoReturn, Optional, Dict, Union

import ray
from ray import ObjectRef
from ray.actor import ActorHandle
from ray.exceptions import RayActorError

from ts_benchmark.utils.parallel.base import TaskResult, SharedStorage

logger = logging.getLogger(__name__)


def is_actor() -> bool:
    """
    判断当前代码是否正在 Ray Actor 中运行

    Ray 有两种运行模式：
    - 主进程模式：worker.mode == 0（可以创建 Actor、调度任务）
    - Actor 模式：worker.mode == 2（只能执行被分配的任务）

    某些操作（如 put 共享存储）只能由主进程执行，
    Actor 内部调用会报错，所以需要这个判断。
    """
    return ray.get_runtime_context().worker.mode == ray.WORKER_MODE


class RayActor:
    """
    单个 Ray 工作进程

    每个 RayActor 是一个独立的操作系统进程，拥有自己的：
    - Python 解释器
    - CPU 资源
    - GPU 资源（可选）

    工作流程：
    1. 初始化：执行 initializers 中的函数（如设置随机种子）
    2. 待命：idle=True，等待任务分配
    3. 执行：收到 run(fn, args) → 执行 fn(*args) → 返回结果
    4. 返回待命状态：idle=True

    关键属性：
    - _idle: True 表示空闲，可以接新任务
    - _start_time: 当前任务的开始时间（用于超时判断）
    - run() 是远程方法，由 RayActorPool 通过 .remote() 调用
    """

    def __init__(self, env: Dict, initializers: Optional[List[Callable]] = None):
        self._idle = True  # 初始状态：空闲
        self._start_time = None  # 当前任务开始时间
        # 执行初始化函数列表（如设置随机种子、注册日志等）
        if initializers is not None:
            for func in initializers:
                func(env)

    def run(self, fn: Callable, args: Tuple) -> Any:
        """
        执行一个任务

        参数：
            fn: 要执行的函数
            args: 函数的参数元组

        这是 Ray 远程方法（被 ActorPool 以 .remote() 调用）。
        记录开始时间（用于超时检查），执行后返回结果。
        """
        self._start_time = time.time()  # 记录任务开始时间
        self._idle = False  # 标记为忙碌
        res = fn(*args)  # 执行任务
        self._idle = True  # 标记为空闲
        return res

    def start_time(self) -> Optional[float]:
        """
        获取当前任务的开始时间

        如果 Actor 空闲或还没开始过任务，返回 None。
        用于超时判断：主进程定期轮询每个 Actor 的开始时间，
        如果 (当前时间 - 开始时间) > timeout，则超时。
        """
        return None if self._idle or self._start_time is None else self._start_time


@ray.remote(max_restarts=-1)
class ObjectRefStorageActor:
    """
    共享存储的 Ray Actor 实现 —— 跨进程的数据共享中心

    这是一个特殊的 Ray Actor，它的作用是存放 Ray ObjectRef。
    什么是 ObjectRef？Ray 中的"远程指针"——指向一个在 Ray
    集群中某处存储的对象。多个进程可以通过 ObjectRef 访问
    同一个数据，而不需要复制。

    为什么需要这个 Actor？
      Ray 的 ObjectRef 存放不直接支持进程间共享。
      这个 Actor 作为"中间人"，存储 name → ObjectRef 的映射。
      主进程 put 数据时，先 ray.put(value) 得到 ObjectRef，
      再把这个 ObjectRef 存入 StorageActor。
      其他进程 get 时，从 StorageActor 拿到 ObjectRef，
      再 ray.get(obj_ref) 取回数据。

    max_restarts=-1：Actor 崩溃后自动无限重启，保证数据不丢失。
    """

    def __init__(self):
        self.storage = {}  # 键值对存储：name → ObjectRef

    def put(self, name: str, value: List[ObjectRef]) -> NoReturn:
        """存储一组 ObjectRef（以列表形式存放）"""
        self.storage[name] = value

    def get(self, name: str) -> Optional[List[ObjectRef]]:
        """取出之前存储的 ObjectRef 列表"""
        return self.storage.get(name)


class RaySharedStorage(SharedStorage):
    """
    Ray 共享存储 —— 对 ObjectRefStorageActor 的封装

    提供两个核心操作：
    - put(name, value)：将 Python 对象存入共享存储
      1. 用 ray.put(value) 把数据变成 ObjectRef
      2. 把 ObjectRef 存入 StorageActor
    - get(name)：从共享存储取回 Python 对象
      1. 从 StorageActor 取出 ObjectRef
      2. 用 ray.get(obj_ref) 反序列化回原对象

    注意：put 必须在主进程中调用（不能在 Actor 中调用），
    因为只有主进程有权限修改共享存储。
    """

    def __init__(self, object_ref_actor):
        self.object_ref_actor = object_ref_actor  # ObjectRefStorageActor 的句柄

    def put(self, name: str, value: Any) -> NoReturn:
        """
        将数据存入共享存储（仅主进程可调用）

        流程：
        1. ray.put(value) → 将 Python 对象序列化，存入 Ray 集群
        2. StorageActor.put(name, [obj_ref]) → 把 ObjectRef 存起来
        """
        if is_actor():
            raise RuntimeError("put is not supported to be called by actors")
        obj_ref = ray.put(value)  # 将数据放入 Ray 的分布式存储
        ray.get(self.object_ref_actor.put.remote(name, [obj_ref]))  # 存 ObjectRef

    def get(self, name: str, default_value: Any = None) -> Any:
        """
        从共享存储取出数据

        流程：
        1. StorageActor.get(name) → 拿到 ObjectRef 列表
        2. ray.get(obj_ref[0]) → 反序列化取回原对象

        如果不存在，返回 default_value
        """
        obj_ref = ray.get(self.object_ref_actor.get.remote(name))  # 获取 ObjectRef
        if obj_ref is None:
            logger.info("data '%s' does not exist in shared storage", name)
            return default_value
        return ray.get(obj_ref[0])  # 从 ObjectRef 还原数据


class RayResult(TaskResult):
    """
    Ray 异步任务的结果句柄

    基于 threading.Event 实现阻塞等待：
    - 调用 result() 时，如果任务还没完成，Event.wait() 会阻塞
    - 任务完成后，主循环线程调用 put() → Event.set() → result() 解除阻塞

    类比：一部对讲机（Event），
    - 主线程等待：按着对讲机，等工人说"做完了"
    - 任务完成时：工人按对讲机说"做完了" → 主线程拿到结果

    特殊处理：如果存入的是 Exception，result() 会 raise 该异常。
    这样异常可以跨线程传递。
    """

    __slots__ = ["_event", "_result"]  # 用 __slots__ 节省内存（有很多任务）

    def __init__(self, event: threading.Event):
        self._event = event  # 线程同步事件，像"任务完成"的信号灯
        self._result = None  # 任务的结果值

    def put(self, value: Any) -> NoReturn:
        """
        存入结果并唤醒等待者
        由主循环线程在任务完成（或超时/崩溃）后调用
        """
        self._result = value
        self._event.set()  # 点亮信号灯 → 唤醒所有等待 .result() 的线程

    def result(self) -> Any:
        """
        阻塞等待，直到任务完成
        如果结果是 Exception 类型，则 raise 该异常
        """
        self._event.wait()  # 阻塞，直到信号灯亮起
        if isinstance(self._result, Exception):
            raise self._result  # 将异常传递回调用方
        else:
            return self._result


class RayTask:
    """
    任务的元信息（内部数据结构）

    存储一个正在执行的任务的完整信息：
    - result: RayResult 对象（任务结果句柄）
    - actor_id: 执行该任务的 Actor 编号
    - timeout: 超时时间（秒），-1 表示永不过期
    - start_time: 任务实际开始时间（从 Actor 拉取）
    """

    __slots__ = ["result", "actor_id", "timeout", "start_time"]

    def __init__(
        self, result: Any = None, actor_id: Optional[int] = None, timeout: float = -1
    ):
        self.result = result  # RayResult 对象
        self.actor_id = actor_id  # 执行任务的 Actor 编号
        self.timeout = timeout  # 超时时间
        self.start_time = None  # 任务实际开始时间（稍后填充）


class RayActorPool:
    """
    Ray Actor 资源池 —— 并行任务调度引擎

    这是整个并行框架的核心！管理一组 RayActor（工作进程），
    负责任务分配、超时检测、Worker 重启等。

    与 Ray 内置 ActorPool 的区别：
    - Ray 内置版不支持每个任务的超时限制
    - 本实现专门为每个任务设了 timeout，超时自动杀掉 Actor 重启

    工作原理（事件驱动 + 后台线程）：
    ┌─────────────────────────────────────────┐
    │  schedule(fn, args) → 放到等待队列      │
    │  返回 RayResult（结果句柄）             │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  后台线程 _main_loop（永久循环）        │
    │                                        │
    │  每轮循环：                            │
    │  1. 检查重启中的 Actor 是否就绪        │
    │  2. 检查活跃任务是否完成/超时          │
    │  3. 有空闲 Actor + 等待队列有任务      │
    │     → 分配任务给空闲 Actor             │
    └─────────────────────────────────────────┘

    线程安全设计：
    - _pending_queue: 线程安全的 Queue，主线程放入，Loop 线程取出
    - _task_info / _active_tasks 等：仅 Loop 线程访问
    - _idle_event: 用于主线程等待所有任务完成

    关键概念：
    - max_tasks_per_child: 每个 Actor 执行 N 个任务后自动重启
      防止内存泄漏或 GPU 状态累积
    - _restarting_actor_pool: 正在重启的 Actor 集合
      重启有 5 秒冷却时间，防止频繁重启
    """

    def __init__(
        self,
        n_workers: int,
        env: Dict,
        per_worker_resources: Optional[Dict] = None,
        max_tasks_per_child: Optional[int] = None,
        worker_initializers: Optional[List[Callable]] = None,
    ):
        if per_worker_resources is None:
            per_worker_resources = {}

        self.env = env  # 环境信息（含共享存储等）
        self.per_worker_resources = per_worker_resources  # 每个 Worker 的资源配置
        self.max_tasks_per_child = max_tasks_per_child  # 每个 Actor 最多执行几个任务
        self.worker_initializers = worker_initializers  # Worker 初始化函数列表

        # 创建 Ray Actor 类（装饰器：指定 CPU/GPU 资源）
        self.actor_class = ray.remote(
            max_restarts=0,  # 不允许自动重启（我们自己管理重启）
            num_cpus=per_worker_resources.get("num_cpus", 1),
            num_gpus=per_worker_resources.get("num_gpus", 0),
        )(RayActor)

        # 创建 n_workers 个 Actor
        self.actors = [self._new_actor() for _ in range(n_workers)]

        # ===== 以下数据仅在主线程中访问 =====
        self._task_counter = itertools.count()  # 任务 ID 自增计数器

        # ===== 以下数据仅在 Loop 线程中访问 =====
        self._task_info = {}  # task_id → RayTask 的映射
        self._ray_task_to_id = {}  # Ray 任务对象 → task_id 的反向映射
        self._active_tasks = []  # 当前活跃（未完成）的任务列表
        self._idle_actors = list(range(len(self.actors)))  # 空闲 Actor 的编号列表
        self._restarting_actor_pool = {}  # 正在重启的 Actor：id → 重启开始时间
        self._actor_tasks = [0] * len(self.actors)  # 每个 Actor 已执行的任务数

        # ===== 线程间通信 =====
        self._is_closed = False  # 是否已关闭
        self._idle_event = threading.Event()  # 空闲事件：所有任务完成后 set
        self._pending_queue = queue.Queue(maxsize=1000000)  # 待调度任务队列

        # 启动后台主循环线程
        self._main_loop_thread = threading.Thread(target=self._main_loop)
        self._main_loop_thread.start()

    def _new_actor(self) -> ActorHandle:
        """
        创建一个新的 Ray Actor

        Windows 特殊处理：
        当使用 GPU 时，Ray 在 Windows 上有 bug，
        可能导致并发计数器错乱。临时方案是把 max_concurrency
        设大，避免问题（代价是可能有额外开销）。

        返回：ActorHandle（可以远程调用该 Actor 的句柄）
        """
        # Windows GPU 模式下的临时规避
        max_concurrency = (
            100
            if sys.platform == "win32"
            and self.per_worker_resources.get("num_gpus", 0) > 0
            else 2
        )
        handle = self.actor_class.options(max_concurrency=max_concurrency).remote(
            self.env,
            self.worker_initializers,
        )

        return handle

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        """
        调度一个任务（主线程调用，非阻塞）

        流程：
        1. 清除 idle_event（表示有新任务）
        2. 生成唯一的 task_id
        3. 创建 RayResult 句柄
        4. 把任务放入等待队列
        5. 立即返回 RayResult（后台线程会处理分配）

        参数：
            fn: 要执行的函数
            args: 函数参数元组
            timeout: 超时时间（秒），-1 表示不超时

        返回：
            RayResult: 结果句柄，调用 .result() 等待结果
        """
        self._idle_event.clear()  # 清除空闲标志
        task_id = next(self._task_counter)  # 分配任务 ID
        result = RayResult(threading.Event())  # 创建结果句柄
        self._pending_queue.put((fn, args, timeout, task_id, result), block=True)  # 入队
        return result  # 立即返回结果句柄

    def _handle_ready_tasks(self, tasks: List) -> NoReturn:
        """
        处理已完成的任务

        对每个完成的任务：
        1. 从 Ray 获取结果
        2. 把结果存入 RayResult（唤醒等待的线程）
        3. 检查是否达到 max_tasks_per_child
        4. 如果需要重启 Actor 则重启，否则将其标记为空闲

        异常处理：
        - RayActorError：Actor 崩溃 → 存入 RuntimeError → 重启 Actor
        """
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            try:
                task_info.result.put(ray.get(task_obj))  # 获取结果并写入句柄
            except RayActorError as e:
                # Actor 崩溃：记录错误并重启
                logger.info(
                    "task %d died unexpectedly on actor %d: %s",
                    task_id,
                    task_info.actor_id,
                    e,
                )
                task_info.result.put(RuntimeError(f"task died unexpectedly: {e}"))
                self._restart_actor(task_info.actor_id)
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
                continue

            # Actor 任务计数 +1
            self._actor_tasks[task_info.actor_id] += 1

            # 检查是否需要重启（达到任务数上限）
            if (
                self.max_tasks_per_child is not None
                and self._actor_tasks[task_info.actor_id] >= self.max_tasks_per_child
            ):
                logger.info(
                    "max_tasks_per_child reached in actor %s, restarting",
                    task_info.actor_id,
                )
                self._restart_actor(task_info.actor_id)  # 重启释放资源
            else:
                self._idle_actors.append(task_info.actor_id)  # 标记为空闲

            # 清理任务记录
            del self._task_info[task_id]
            del self._ray_task_to_id[task_obj]

    def _get_duration(self, task_info: RayTask) -> Optional[float]:
        """
        获取任务已执行的时长

        首次调用时从 Actor 拉取 start_time（远程调用）。
        - 返回 None：Actor 已崩溃
        - 返回 -1：start_time 为 None（任务还没开始）
        - 返回正数：已执行秒数

        这个函数是超时检测的核心。
        """
        if task_info.start_time is None:
            try:
                task_info.start_time = ray.get(
                    self.actors[task_info.actor_id].start_time.remote()
                )
            except RayActorError as e:
                logger.info(
                    "actor %d died unexpectedly: %s, restarting...",
                    task_info.actor_id,
                    e,
                )
                return None

        return (
            -1 if task_info.start_time is None else time.time() - task_info.start_time
        )

    def _handle_unfinished_tasks(self, tasks: List) -> NoReturn:
        """
        检查未完成的任务是否超时

        遍历所有活跃任务：
        - 如果执行时长超过 timeout → 杀掉 Actor + 返回 TimeoutError
        - 否则保留在活跃列表

        超时处理：直接重启 Actor（kill + 创建新的），
        不尝试恢复（因为不清楚任务卡在哪里）。
        """
        new_active_tasks = []
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            duration = self._get_duration(task_info)

            if duration is None or 0 < task_info.timeout < duration:
                if duration is not None:
                    logger.info(
                        "actor %d killed after timeout %f",
                        task_info.actor_id,
                        task_info.timeout,
                    )

                self._restart_actor(task_info.actor_id)  # 重启 Actor

                # 通知等待者：任务超时
                task_info.result.put(
                    TimeoutError(f"time limit exceeded: {task_info.timeout}")
                )

                # 清理记录
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
            else:
                new_active_tasks.append(task_obj)  # 保留此任务

        self._active_tasks = new_active_tasks

    def _restart_actor(self, actor_id: int) -> NoReturn:
        """
        重启一个 Actor

        步骤：
        1. ray.kill 杀掉旧 Actor（no_restart=True 防止自动复活）
        2. 创建新 Actor 替换
        3. 重置任务计数器
        4. 记录重启时间（用于冷却检查）

        注意：Ray 旧 Actor 的析构可能不是立即可用的，
        所以新 Actor 需要冷却时间等待它就绪。
        """
        cur_actor = self.actors[actor_id]
        ray.kill(cur_actor, no_restart=True)  # 杀掉旧 Actor
        del cur_actor

        self.actors[actor_id] = self._new_actor()  # 创建新 Actor
        self._actor_tasks[actor_id] = 0  # 重置任务计数
        self._restarting_actor_pool[actor_id] = time.time()  # 记录重启时间

    def _check_restarting_actors(self):
        """
        检查重启中的 Actor 是否已就绪

        冷却时间 5 秒后才检查（给新 Actor 启动的时间）。
        通过调用 start_time.remote() 来测试 Actor 是否响应。
        如果响应了 → 加入空闲列表。
        """
        new_restarting_pool = {}
        for actor_id, restart_time in self._restarting_actor_pool.items():
            # 冷却时间 5 秒
            if time.time() - restart_time > 5:
                # 测试新 Actor 是否可用
                ready_tasks = ray.wait(
                    [self.actors[actor_id].start_time.remote()], timeout=0.5
                )[0]
                if ready_tasks:
                    logger.debug("restarted actor %d is now ready", actor_id)
                    self._idle_actors.append(actor_id)  # 就绪，加入空闲列表
                    continue  # 不放入 new_restarting_pool
                else:
                    logger.debug(
                        "restarted actor %d is not ready, resetting timer", actor_id
                    )
                    self._restarting_actor_pool[actor_id] = time.time()  # 重置计时器

            new_restarting_pool[actor_id] = restart_time
        self._restarting_actor_pool = new_restarting_pool

    def _main_loop(self) -> NoReturn:
        """
        后台主循环（在独立线程中运行）

        每轮循环做三件事：
        1. 检查重启中的 Actor 是否就绪
        2. 检查活跃任务状态（完成/超时）
        3. 如果有空闲 Actor 和待调度任务 → 分配任务

        空闲判断：
        - 没有活跃任务 + 等待队列为空 → 设置 idle_event → 休眠 1 秒
        """
        while not self._is_closed:
            self._check_restarting_actors()  # 步骤 1：检查重启

            logger.debug(
                "%d active tasks, %d idle actors, %d restarting actors",
                len(self._active_tasks),
                len(self._idle_actors),
                len(self._restarting_actor_pool),
            )

            # 空闲状态：没有任务 → 通知并休眠
            if not self._active_tasks and self._pending_queue.empty():
                self._idle_event.set()  # 发"空闲"信号
                time.sleep(1)
                continue

            # 步骤 2：检查活跃任务
            if self._active_tasks:
                ready_tasks, unfinished_tasks = ray.wait(self._active_tasks, timeout=1)
                self._handle_ready_tasks(ready_tasks)
                self._handle_unfinished_tasks(unfinished_tasks)
            else:
                time.sleep(1)

            # 步骤 3：分配新任务
            while self._idle_actors and not self._pending_queue.empty():
                fn, args, timeout, task_id, result = self._pending_queue.get_nowait()
                cur_actor = self._idle_actors.pop()  # 从空闲池取出一个 Actor

                # 将任务提交给该 Actor（远程调用）
                task_obj = self.actors[cur_actor].run.remote(fn, args)

                # 记录任务信息
                self._task_info[task_id] = RayTask(
                    result=result, actor_id=cur_actor, timeout=timeout
                )
                self._ray_task_to_id[task_obj] = task_id
                self._active_tasks.append(task_obj)

                logger.debug("task %d assigned to actor %d", task_id, cur_actor)

    def wait(self) -> NoReturn:
        """
        阻塞直到所有任务完成

        等待条件：等待队列空 + 所有活跃任务完成
        如果还有 Actor 在重启 → 等待它们就绪
        """
        if self._is_closed:
            return
        if self._pending_queue.empty() and not self._active_tasks:
            return
        self._idle_event.clear()
        self._idle_event.wait()  # 等待空闲信号
        # 等待重启中的 Actor 就绪
        while self._restarting_actor_pool:
            time.sleep(1)

    def close(self) -> NoReturn:
        """
        关闭 Actor 池

        1. 标记关闭
        2. 杀掉所有 Actor
        3. 等待后台线程结束
        """
        self._is_closed = True
        for actor in self.actors:
            ray.kill(actor)  # 杀掉所有 Actor
        self._main_loop_thread.join()  # 等待 Loop 线程退出


class RayBackend:
    """
    Ray 并行后端 —— 顶层统一接口

    这是用户代码直接使用的入口，封装了：
    - Ray 初始化
    - CPU/GPU 资源分配
    - 共享存储创建
    - ActorPool 管理

    用法示例：
        backend = RayBackend(
            n_workers=4,          # 4 个 Worker 进程
            n_cpus=4,            # 使用 4 个 CPU 核心
            gpu_devices=[0, 1],  # 使用 GPU 0 和 GPU 1
        )
        backend.init()

        # 提交任务
        result = backend.schedule(my_train_func, (model, data))

        # 等待结果
        accuracy = result.result()

        backend.close()

    资源分配策略：
    - CPU：尽量均匀分配，除不尽时取整
      → 如 10 个 CPU、4 个 Worker → 每个 2 个 CPU（剩余 2 个不用）
    - GPU：与 CPU 类似，各 GPU 可均匀分配到不同 Worker
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        n_cpus: Optional[int] = None,
        gpu_devices: Optional[List[int]] = None,
        max_tasks_per_child: Optional[int] = None,
        worker_initializers: Optional[Union[List[Callable], Callable]] = None,
    ):
        # CPU 数：未指定则取系统 CPU 核数
        self.n_cpus = n_cpus if n_cpus is not None else os.cpu_count()
        # Worker 数：未指定则等于 CPU 数（一般建议 worker ≤ CPU）
        self.n_workers = n_workers if n_workers is not None else self.n_cpus
        # GPU 设备列表：如 [0, 2] 表示使用 GPU 0 和 2
        self.gpu_devices = gpu_devices if gpu_devices is not None else []
        # 每个 Actor 执行多少个任务后重启
        self.max_tasks_per_child = max_tasks_per_child
        # Worker 初始化函数（统一为列表格式）
        self.worker_initializers = (
            worker_initializers
            if isinstance(worker_initializers, list)
            else [worker_initializers]
        )
        self.pool = None  # RayActorPool 实例（init 后创建）
        self._storage = None  # 共享存储实例
        self.initialized = False  # 是否已初始化

    def init(self) -> NoReturn:
        """
        初始化 Ray 后端（必须最先调用）

        流程：
        1. 计算每个 Worker 的 CPU/GPU 配额
        2. 初始化 Ray（如果尚未初始化）
        3. 创建共享存储
        4. 创建 ActorPool

        注意：不能在 Ray Actor 中调用 init。
        """
        if self.initialized:
            return

        # 计算每个 Worker 的 CPU 配额
        cpu_per_worker = self._get_cpus_per_worker(self.n_cpus, self.n_workers)
        # 计算每个 Worker 的 GPU 配额
        gpu_per_worker, gpu_devices = self._get_gpus_per_worker(
            self.gpu_devices, self.n_workers
        )

        # 初始化 Ray（主进程）
        if not ray.is_initialized():
            # 设置可见 GPU 设备
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
            ray.init(num_cpus=self.n_cpus, num_gpus=len(gpu_devices))
        else:
            raise RuntimeError("init is not allowed to be called in ray actors")

        # 创建共享存储（put 操作只能由主进程执行）
        self._storage = RaySharedStorage(ObjectRefStorageActor.remote())

        # 创建 Actor 池
        self.pool = RayActorPool(
            self.n_workers,
            self.env,
            {
                "num_cpus": cpu_per_worker,
                "num_gpus": gpu_per_worker,
            },
            max_tasks_per_child=self.max_tasks_per_child,
            worker_initializers=self.worker_initializers,
        )
        self.initialized = True

    def add_worker_initializer(self, func: Callable) -> NoReturn:
        """
        添加 Worker 初始化函数

        所有 Worker（新创建的和正在重启的）都会执行这些初始化函数。
        典型用途：注册自定义模块、设置随机种子等。
        """
        if self.worker_initializers is None:
            self.worker_initializers = []
        self.worker_initializers.append(func)
        self.pool.worker_initializers = self.worker_initializers

    @property
    def env(self) -> Dict:
        """获取环境信息（供任务使用的上下文）"""
        return {
            "storage": self.shared_storage,
        }

    def _get_cpus_per_worker(self, n_cpus: int, n_workers: int) -> Union[int, float]:
        """
        计算每个 Worker 分多少 CPU

        策略：
        - 如果 CPU 数能整除 Worker 数 → 精确平分
        - 如果不能整除 → 取整，舍弃余数（因为 Ray 不容许超分）
        """
        if n_cpus > n_workers and n_cpus % n_workers != 0:
            cpus_per_worker = n_cpus // n_workers
            logger.info(
                "only %d among %d cpus are used to match the number of workers",
                cpus_per_worker * n_workers,
                n_cpus,
            )
        else:
            cpus_per_worker = n_cpus / n_workers
        return cpus_per_worker

    def _get_gpus_per_worker(
        self, gpu_devices: List[int], n_workers: int
    ) -> Tuple[Union[int, float], List[int]]:
        """
        计算每个 Worker 分多少 GPU

        策略与 CPU 类似：尽量均匀分配。
        返回 (每 Worker 的 GPU 数, 实际使用的 GPU 设备列表)
        """
        n_gpus = len(gpu_devices)
        if n_gpus > n_workers and n_gpus % n_workers != 0:
            gpus_per_worker = n_gpus // n_workers
            used_gpu_devices = gpu_devices[: gpus_per_worker * n_workers]
            logger.info(
                "only %s gpus are used to match the number of workers", used_gpu_devices
            )
        else:
            gpus_per_worker = n_gpus / n_workers
            used_gpu_devices = gpu_devices
        return gpus_per_worker, used_gpu_devices

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        """
        调度一个任务（并行执行）

        与 SequentialBackend 的 schedule 接口完全一致。
        参数：
            fn: 要执行的函数
            args: 函数参数元组
            timeout: 超时时间（秒），-1 不限制
        返回：
            RayResult: 调用 .result() 阻塞等待结果
        """
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")
        return self.pool.schedule(fn, args, timeout)

    def close(self, force: bool = False) -> NoReturn:
        """
        关闭后端

        参数：
            force: False 则等待所有任务完成后再关闭；
                   True 则直接关闭（任务可能丢失）
        """
        if not self.initialized:
            return

        if not force:
            self.pool.wait()  # 等所有任务完成
        self.pool.close()  # 关闭 Actor 池
        ray.shutdown()  # 关闭 Ray

    @property
    def shared_storage(self) -> RaySharedStorage:
        """获取共享存储实例"""
        return self._storage

    def execute_on_workers(self, func: Callable) -> NoReturn:
        """
        在所有 Worker 上同步执行一个函数

        只在池空闲时可用。会等待所有 Worker 都执行完毕。
        用途：向所有 Worker 广播一个操作（如加载数据）。
        """
        self.pool.wait()  # 确保池空闲

        # 广播到所有 Actor 执行
        tasks = []
        for actor in self.pool.actors:
            tasks.append(actor.run.remote(func, (self.env,)))
        ray.wait(tasks, num_returns=len(tasks))  # 等所有完成


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backend = RayBackend(3, max_tasks_per_child=1)
    backend.init()

    def sleep_func(t):
        time.sleep(t)
        print(f"sleep after {t}")
        return t

    results = []
    results.append(backend.schedule(sleep_func, (10,), timeout=5))
    results.append(backend.schedule(sleep_func, (10,), timeout=20))
    results.append(backend.schedule(sleep_func, (1,), timeout=5))
    results.append(backend.schedule(sleep_func, (2,), timeout=5))
    results.append(backend.schedule(sleep_func, (3,), timeout=5))
    results.append(backend.schedule(sleep_func, (4,), timeout=5))
    results.append(backend.schedule(sleep_func, (5,), timeout=5))
    results.append(backend.schedule(sleep_func, (6,), timeout=5))

    for i, res in enumerate(results):
        try:
            print(f"{i}-th task result: {res.result()}")
        except TimeoutError:
            print(f"{i}-th task fails after timeout")

    backend.close()
    # time.sleep(100)
    # time.sleep(1)
    # pool.wait()
    # pool.close()
