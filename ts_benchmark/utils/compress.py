# -*- coding: utf-8 -*-
"""
压缩/解压缩工具模块
===================

负责实验结果的压缩存储和读取解压。

为什么需要压缩？
  跑一次实验可能产生几十上百条评估结果，每条结果包含多列指标。
  不压缩的话，结果文件会越来越多，占存储空间。
  压缩后可以：
    - 多个CSV文件打包成一个 tar.gz，方便传输
    - 节省磁盘空间（gzip/tar.gz 压缩率很高）

类比理解：
  就像实验报告归档——把所有实验记录纸装订成册（tar），
  然后压真空袋保存（gzip），要用时再拆包（decompress）。

支持的压缩格式：
  - tar.gz：默认格式，支持多个文件打包 + 压缩
  - gzip：单文件压缩（目前兼容但非默认）
"""

from __future__ import absolute_import

import gzip
import tarfile
from io import BytesIO
from typing import Dict, Optional


def compress_gz(data: Dict[str, str]) -> bytes:
    """
    以 tar.gz 格式打包并压缩数据

    当你要把多个CSV文件（如多次实验记录）打包成
    一个 .tar.gz 文件时使用。

    工作原理：
      1. 创建一个 tar 归档
      2. 把每个文件（key=文件名, value=内容）加入归档
      3. 用 gzip 压缩

    参数：data → {"文件名": "文件文本内容"}
    返回：二进制的 .tar.gz 数据
    """
    outbuf = BytesIO()  # 内存缓冲区，避免写磁盘

    with tarfile.open(fileobj=outbuf, mode="w:gz") as tar:
        for k, v in data.items():
            info = tarfile.TarInfo(name=k)     # 创建文件信息头
            v_bytes = v.encode("utf8")         # 文本转字节
            info.size = len(v_bytes)           # 记录文件大小
            tar.addfile(info, fileobj=BytesIO(v_bytes))  # 添加进归档

    return outbuf.getvalue()  # 返回压缩后的二进制数据


def compress_gzip(data: Dict[str, str]) -> bytes:
    """
    以 gzip 格式压缩数据（目前仅作备用）

    注意：gzip 不支持多个文件打包，所以会把多个
    key=value 拼成一行行文本再压缩。
    """
    outbuf = BytesIO()

    with gzip.GzipFile(fileobj=outbuf, mode="wb") as gz:
        for k, v in data.items():
            v_bytes = v.encode("utf8")
            gz.write(v_bytes)

    return outbuf.getvalue()


def decompress_gzip(compressed_data: bytes) -> Dict[str, str]:
    """
    解压 gzip 格式的数据

    对应 compress_gzip 的逆操作。
    """
    decompressed_data = {}
    compressed_buf = BytesIO(compressed_data)

    with gzip.GzipFile(fileobj=compressed_buf, mode="rb") as gz:
        while True:
            chunk = gz.read(1024)  # 每次读1KB
            if not chunk:
                break
            chunk_str = chunk.decode("utf8")
            key_values = chunk_str.split("\n")
            for key_value in key_values:
                if key_value:
                    key, value = key_value.split(":")
                    decompressed_data[key] = value

    return decompressed_data


def decompress_gz(data: bytes) -> Dict[str, str]:
    """
    解压 tar.gz 格式的数据

    这是最常用的解压方式，对应 compress_gz 的逆操作。
    会把打包的多个文件还原为 {"文件名": "内容"} 的字典。

    参数：data → 压缩后的二进制数据
    返回：{"文件名": "文本内容"}
    """
    ret = {}
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():        # 遍历归档中的每个文件
            if member.isfile():                # 只处理文件（跳过目录）
                # 读取文件内容并解码为文本
                ret[member.name] = tar.extractfile(member).read().decode("utf8")
    return ret


def compress(data: Dict[str, str], method: str = "gz") -> bytes:
    """
    统一的压缩入口函数

    根据 method 参数选择对应的压缩算法。

    参数：
        data:   待压缩的文件字典
        method: 压缩方式，默认 "gz"（即 tar.gz）
    返回：压缩后的二进制数据
    """
    if method != "gz":
        compress_gzip(data)
    return compress_gz(data)


def decompress(data: bytes, method: str = "gz") -> Dict[str, str]:
    """
    统一的解压入口函数

    参数：
        data:   压缩后的二进制数据
        method: 压缩方式，默认 "gz"
    返回：解压后的文件字典
    """
    if method != "gz":
        decompress_gzip(data)
    return decompress_gz(data)


def get_compress_file_ext(method: str) -> str:
    """
    根据压缩方式返回对应的文件扩展名

    例如：method="gz" → 返回 "tar.gz"
    这样在保存文件时可以自动追加正确的扩展名。

    参数：method 压缩方式
    返回：文件扩展名字符串
    """
    if method != "gz":
        return "gzip"
    return "tar.gz"


def get_compress_method_from_ext(ext: str) -> Optional[str]:
    """
    根据文件扩展名反查压缩方式

    例如：ext="tar.gz" → 返回 "gz"
    如果扩展名不匹配任何已知格式 → 返回 None

    用途：读取文件时自动判断该如何解压
    """
    return {
        "tar.gz": "gz"
    }.get(ext)
