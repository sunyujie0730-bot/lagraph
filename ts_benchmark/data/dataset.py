# -*- coding: utf-8 -*-
"""
数据集模块（Dataset）
====================
Dataset 是框架内部的数据"仓库"，用一个对象管理所有数据信息。

它就像一个"保险柜"，里面有三层抽屉：
  - 数据抽屉（_data_dict）:    存放所有时序 DataFrames，key 是文件名
  - 协变量抽屉（_covariate_dict）: 存放辅助信息（天气、节假日等外部因素）
  - 元数据抽屉（_metadata）:    存放每个文件的属性表（DataFrame 格式）

为什么需要 Dataset 而不是直接用 dict？
  1. 封装了数据校验逻辑（虽然目前校验比较松）
  2. 提供统一的增量更新接口（update_data）
  3. 支持序列化和恢复状态（get_state / set_state）
  4. 方便后续扩展更严格的校验规则

典型用法：
  ds = Dataset()
  ds.set_data(data_dict={"MSL.csv": df}, metadata=meta_df)
  ds.update_data(inc_data_dict={"MSL.csv": df2})
  series = ds.get_series("MSL.csv")   # 获取单个时间序列
  info = ds.get_series_meta_info("MSL.csv")  # 获取元信息
"""
from typing import Optional, Dict, NoReturn

import pandas as pd


class Dataset:
    """
    数据集类：存储时序数据和元信息

    核心职责：
    1. 维护数据字典、协变量字典、元数据表三个核心容器
    2. 提供 set_data / update_data 两种更新方式
    3. 提供 get_series / get_covariates / get_series_meta_info 三种查询方式
    4. 支持状态序列化（方便在 Ray 多进程间共享数据）

    数据一致性约定：
    - data_dict 的 key 应和 metadata 的 index 保持一致
    - 目前 _validate_data 不强制检查，但预留了扩展空间
    """

    def __init__(self):
        """初始化三个空容器"""
        self._metadata = None           # 元数据表（pd.DataFrame）
        self._data_dict = {}            # 时序数据字典 {文件名: DataFrame}
        self._covariate_dict = {}       # 协变量字典 {文件名: 协变量信息}

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """
        返回完整的元数据表（DataFrame 格式）

        ⚠️ 注意：不要在返回值上进行原地修改（inplace）
        返回的是内部引用，修改会直接影响 Dataset 内部状态，
        可能导致数据不一致。
        """
        return self._metadata

    def set_data(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> NoReturn:
        """
        整体设置数据集（全量替换模式）

        通常用于初始化阶段，一次性传入所有数据。
        如果某个参数传 None，该部分保持旧值不变。

        参数：
            data_dict:      时序数据字典 {文件名: DataFrame}
                           DataFrame 格式需符合 OTB 协议
                           传 None 表示不修改
            covariate_dict: 协变量字典 {文件名: 协变量信息}
                           传 None 表示不修改
            metadata:       元数据表（DataFrame），索引为文件名
                           传 None 表示不修改
        """
        # 如果传了新的就替换，否则保留旧的
        new_metadata = metadata if metadata is not None else self._metadata
        new_data_dict = data_dict if data_dict is not None else self._data_dict
        new_covariate_dict = (
            covariate_dict if covariate_dict is not None else self._covariate_dict
        )

        # 校验数据一致性（目前为空实现，但预留了扩展点）
        self._validate_data(new_data_dict, new_covariate_dict, new_metadata)

        # 更新内部状态
        self._metadata = new_metadata
        self._data_dict = new_data_dict
        self._covariate_dict = new_covariate_dict

    def _validate_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        covariate_dict: Dict[str, Dict],
        metadata: Optional[pd.DataFrame],
    ) -> NoReturn:
        """
        校验数据字典、协变量和元数据之间的兼容性

        目前为空实现（不强制检查）。
        预留扩展点：未来可以在这里加入以下检查：
        - data_dict 的 key 和 metadata.index 是否匹配
        - 每个 DataFrame 的列数是否和元数据中记录一致
        - 协变量是否和对应时序的时间轴对齐

        参数：
            data_dict:      时序数据字典
            covariate_dict: 协变量字典
            metadata:       元数据表
        """
        pass

    def update_data(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> NoReturn:
        """
        增量更新数据（字典合并模式）

        与 set_data（全量替换）不同，update_data 只在现有数据上追加新条目。
        适合"按需加载"场景：加载了一批文件后又需要加载另一批。

        参数：
            inc_data_dict:      新增的时序数据字典
            inc_covariate_dict: 新增的协变量字典
        """
        self._validate_update_data(inc_data_dict, inc_covariate_dict)
        # 直接 update，已有 key 会被覆盖
        self._data_dict.update(inc_data_dict)
        self._covariate_dict.update(inc_covariate_dict)

    def _validate_update_data(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> NoReturn:
        """
        校验增量数据与现有数据的兼容性

        目前为空实现。预留扩展点：
        - 检查新数据是否和已有元数据匹配
        - 检查新增数据是否与旧数据列结构一致
        """
        pass

    def clear_data(self) -> NoReturn:
        """
        清空所有数据

        释放内存，恢复到初始状态。
        常用于切换数据集或实验后清理。
        """
        self._metadata = None
        self._data_dict = {}
        self._covariate_dict = {}

    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        按名字获取单个时间序列

        参数：
            name: 文件名（如 "MSL.csv"）
        返回：
            对应的 DataFrame，如果不存在则返回 None
        """
        return self._data_dict.get(name, None)

    def get_covariates(self, name: str) -> Optional[Dict]:
        """
        按名字获取协变量（辅助信息）

        协变量通常包含：
        - 时间特征（星期几、是否节假日）
        - 外部数据（天气、舆情等）

        参数：
            name: 文件名
        返回：
            协变量字典，如果不存在则返回 None
        """
        return self._covariate_dict.get(name, None)

    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        获取某个时间序列的元信息

        安全策略：
        - 如果数据字典中没有该序列，返回 None（即使元数据表中有）
        - 这样防止"有元信息但没数据"的不一致状态

        参数：
            name: 文件名
        返回：
            元信息 Series，如果不存在返回 None
        """
        # 双重检查：数据存在 && 元信息存在
        if name not in self._data_dict:
            return None
        if self._metadata is None or name not in self._metadata.index:
            return None
        return self._metadata.loc[name]

    def has_series(self, name: str) -> bool:
        """
        检查某个序列是否已加载

        参数：
            name: 文件名
        返回：
            True 表示数据已加载，False 表示不在数据字典中
        """
        return name in self._data_dict

    def has_series_meta_info(self, name: str) -> bool:
        """
        检查某个序列是否有元信息

        比 has_series 更严格：需要同时满足
        1. 数据已加载
        2. 元数据表存在
        3. 该序列在元数据表中有记录

        参数：
            name: 文件名
        返回：
            True 表示数据+元信息都齐全
        """
        return (
            self.has_series(name)
            and self._metadata is not None
            and name in self._metadata.index
        )

    def get_state(self) -> Dict:
        """
        获取可序列化的状态快照

        用于 Ray 进程间数据传输：
        主进程调用 get_state() 获取快照 → Ray 序列化传输
        → Worker 进程调用 set_state() 恢复

        返回：
            包含 metadata、data_dict、covariate_dict 的字典
        """
        return {
            "metadata": self._metadata,
            "data_dict": self._data_dict,
            "covariate_dict": self._covariate_dict,
        }

    def set_state(self, state: Dict) -> NoReturn:
        """
        从序列化的状态快照恢复

        是 get_state() 的逆操作，配合 Ray 的零拷贝传输使用。

        参数：
            state: get_state() 返回的字典
        """
        self._metadata = state["metadata"]
        self._data_dict = state["data_dict"]
        self._covariate_dict = state["covariate_dict"]
