# -*- coding: utf-8 -*-
"""
文件名生成工具模块

负责生成带有唯一标识的实验结果文件名，
确保多台机器、多个实验同时运行时不会重名。

背景：在一个分布式实验环境中，多台机器可能同时
跑同一个模型，为了避免结果文件互相覆盖，
文件名中需要包含机器、时间、进程的标识信息。
"""

import os
import socket
import time


def get_unique_file_suffix():
    """
    生成唯一的文件后缀（人类可读的时间格式）

    格式：_YYYY-MM-DD_HH-MM-SS_序号.csv

    命名规则：
    1. 日期时间：年-月-日_时-分-秒，如 2026-04-28_14-30-25
       → 按文件名排序即是按时间先后排序，直观可读
    2. 微秒：取当前时间的微秒部分（0-999999）
       → 防止同一秒内连续跑两次重名

    与旧版的区别：
        旧：.1765358359.msi.2102035.csv（无规律的Unix时间戳）
        新：_2026-04-28_14-30-25_345678.csv（一目了然的时间）

    用法示例：
        suffix = get_unique_file_suffix()
        # 返回："_2026-04-28_14-30-25_123456.csv"
        full_path = "LaGraph" + suffix
        # → "LaGraph_2026-04-28_14-30-25_123456.csv"

    返回：
        str: 包含日期时间和微秒的字符串后缀
    """
    from datetime import datetime

    # 获取当前时间，精确到微秒
    now = datetime.now()
    # 格式化：2026-04-28_14-30-25
    date_part = now.strftime("%Y-%m-%d_%H-%M-%S")
    # 微秒部分，补齐6位：000123, 456789
    micro_part = f"{now.microsecond:06d}"

    # 拼接：_日期时间_微秒.csv
    log_filename = f"_{date_part}_{micro_part}.csv"
    return log_filename
