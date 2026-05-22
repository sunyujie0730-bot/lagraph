# -*- coding: utf-8 -*-
"""
结果记录模块（Recording）
=========================
负责基准测试结果文件的读写、压缩、保存。

核心功能：
  1. 读取单个或多个记录文件（CSV格式，支持压缩）
  2. 将评估结果写入记录文件（带唯一后缀防重名）
  3. 支持 gz/tar.gz 等压缩格式以节省存储空间
  4. 批量加载和分析历史实验结果

类比理解：
  就像"实验记录本"：
  - 每次实验结束后，把结果整理成一页纸（CSV行）
  - 写进记录本（CSV文件），标注日期防重名
  - 要分析时翻出所有记录本对比（load_record_data）
  - 旧本子可以压压缩存起来（compress）
"""

from __future__ import absolute_import

import io
import itertools
import logging
import os
import os.path
from io import StringIO
from typing import List, Optional

import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import get_unique_file_suffix

logger = logging.getLogger(__name__)


def read_record_file(fn: str) -> pd.DataFrame:
    """
    读取单个记录文件（支持压缩格式）
    
    智能识别文件格式：
    - .csv 文件 → 直接用 pandas 读取
    - .tar.gz / .gz 等压缩文件 → 先解压再读取
    
    对于 tar.gz 格式（包含多个CSV），会把所有CSV合并成一个DataFrame
    
    参数：fn 文件路径
    返回：DataFrame 格式的实验记录
    """
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        # 普通 CSV 文件，直接读取
        return pd.read_csv(fn)
    else:
        # 压缩文件，先读取二进制数据
        with open(fn, "rb") as fh:
            data = fh.read()
        # 解压（返回 dict：{文件名: 内容字符串}）
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8"))))
        # 合并多个CSV为一个大表
        return pd.concat(ret, axis=0)


def write_record_file(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    写入单个记录文件（支持压缩）
    
    参数：
        result_df:      待保存的实验结果 DataFrame
        file_path:      目标文件路径
        compress_method: 压缩方式，None 表示不压缩，如 "gz"
    
    返回：实际写入的文件路径（可能因压缩而追加了扩展名）
    """
    if compress_method is not None:
        # 先写入内存缓冲区（StringIO）
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        # 压缩
        write_data = compress(
            {os.path.basename(file_path): buf.getvalue()}, method=compress_method
        )
        # 追加压缩扩展名（如 .tar.gz）
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        # 不压缩，直接写CSV（默认 line_terminator=os.linesep，Linux 下即 \n）
        result_df.to_csv(file_path, index=False)

    return file_path


def load_record_data(
    record_files: List[str], drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    从多个记录文件中批量加载实验结果
    
    用途：当你跑了很多次实验后，想一次性加载所有结果做对比分析
    
    智能识别：
    - 如果路径是文件 → 直接加载该文件
    - 如果路径是文件夹 → 自动扫描文件夹内所有CSV/压缩文件并加载
    
    参数：
        record_files: 记录文件/文件夹路径列表
        drop_columns: 加载时要丢弃的列（节省内存，如丢弃日志列）
    
    返回：所有记录合并后的 DataFrame
    
    容错设计：
    - 文件不存在/权限不足/格式错误 → 跳过该文件，记录日志
    - 不会因为个别文件损坏而中断整个加载过程
    """
    # 展开文件夹：把目录路径替换为目录内所有记录文件的路径
    record_files = itertools.chain.from_iterable(
        [
            [fn] if not os.path.isdir(fn) else find_record_files(fn)
            for fn in record_files
        ]
    )

    ret = []
    for fn in record_files:
        logger.info("loading log file %s", fn)
        try:
            cur_record = read_record_file(fn)
            if drop_columns:
                cur_record = cur_record.drop(columns=drop_columns)
            ret.append(cur_record)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # 遇到损坏/不兼容的文件就跳过，不影响整体加载
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def find_record_files(directory: str) -> List[str]:
    """
    递归搜索目录中所有记录文件
    
    识别规则：以 .csv 或 .tar.gz 结尾
    （TODO：当前判断较粗略，未来可改进）
    
    参数：directory 待搜索的目录路径
    返回：找到的记录文件路径列表
    """
    record_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 通过扩展名判断是否为记录文件
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                record_files.append(os.path.join(root, file))
    return record_files


def save_log(
    result_df: pd.DataFrame, save_path, file_prefix: str, compress_method: Optional[str] = None
) -> str:
    """
    保存评估结果到日志文件
    
    这是 pipeline 最后一步调用的函数，将评估结果保存为持久化文件。
    
    参数：
        result_df:       评估结果 DataFrame，包含指标值、模型参数等列
        save_path:       保存的子目录路径（相对于 result/ 文件夹）
        file_prefix:     文件名前缀（通常是模型名），如 "LaGraph"
        compress_method: 压缩方式，默认 "gz"（gzip压缩）
    
    返回：保存后的文件路径
    
    文件名生成规则：
    - 前缀 + 唯一后缀（含时间戳和机器ID）→ 防止重名
    - 如：LaGraph.1765358359.msi.2102035.csv.tar.gz
         ↑前缀   ↑时间戳    ↑机器  ↑进程  ↑扩展名
    
    错误日志输出：
    - 如果结果中有错误信息（log_info列），会输出前3条到控制台
    - 更多错误信息需要在记录文件中查看
    """
    # 输出前几条错误日志到控制台（方便快速发现问题）
    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    # 确定保存路径
    if save_path is not None:
        result_path = (
            os.path.join(ROOT_PATH, "result", save_path)
            if not os.path.isabs(save_path)
            else save_path
        )
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)  # 自动创建目录

    # 生成唯一文件名并保存
    record_filename = file_prefix + get_unique_file_suffix()
    file_path = os.path.join(result_path, record_filename)

    return write_record_file(result_df, file_path, compress_method)
