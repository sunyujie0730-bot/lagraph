# -*- coding: utf-8 -*-
"""
设计模式工具模块

提供常用的设计模式实现，方便框架各层复用。

目前包含：
  - Singleton: 单例模式的元类实现
"""

class Singleton(type):
    """
    单例模式元类

    使用方法：让目标类的 metaclass = Singleton，
    该类就会变成单例——无论实例化多少次，
    始终返回同一个对象。

    为什么需要单例？
      在框架中有一些全局唯一的组件，比如：
      - 数据池（DataPool）：整个实验共享同一份数据
      - 全局存储（GlobalStorage）：所有模型写同一个结果集
      使用单例可以保证这些组件在整个生命周期中只有一个实例，
      避免创建多个副本导致数据不一致。

    用法示例：
        class MyManager(metaclass=Singleton):
            def __init__(self):
                self.data = {}

        a = MyManager()
        b = MyManager()
        assert a is b  # a 和 b 是同一个对象

    原理：
      通过元类的 __call__ 方法拦截实例化过程。
      第一次调用时正常创建实例并缓存到 _instance_dict，
      后续调用直接从缓存中返回。
    """

    _instance_dict = {}  # 存储每个类的唯一实例

    def __call__(cls, *args, **kwargs):
        # 如果该类还没有创建过实例，就创建一个新的
        if cls not in cls._instance_dict:
            cls._instance_dict[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # 返回唯一的实例
        return cls._instance_dict[cls]
