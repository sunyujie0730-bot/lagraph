# -*- coding: utf-8 -*-
"""
数据源模块（Data Source）
=========================
负责从本地文件系统加载时间序列数据，是框架的"数据入口"。

核心类层次：
  DataSource（基类）
    └── LocalDataSource（本地目录数据源）
          ├── LocalForecastingDataSource（预测任务专用）
          ├── LocalStForecastingDataSource（时空预测专用）
          └── LocalAnomalyDetectDataSource（异常检测专用）★最常用

类比理解：
  DataSource 就像一个"图书馆管理员"：
  - 知道所有数据文件存放在哪里（本地路径）
  - 有一个目录索引（metadata元数据表）
  - 能按需取书（load_series_list按需加载）
  - 并行取书效率高（ThreadPoolExecutor多线程并发加载）
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, NoReturn, List

import pandas as pd

from ts_benchmark.common.constant import (
    FORECASTING_DATASET_PATH,
    ANOMALY_DETECT_DATASET_PATH,
    ST_FORECASTING_DATASET_PATH,
)
from ts_benchmark.data.dataset import Dataset
from ts_benchmark.data.utils import load_series_info, read_data, read_covariates

logger = logging.getLogger(__name__)


class DataSource:
    """
    数据源基类：管理数据加载的内部容器
    
    职责：
    1. 维护一个内部的 Dataset 对象（数据字典 + 协变量字典 + 元数据表）
    2. 提供 load_series_list() 接口让子类实现具体的数据加载逻辑
    
    内部结构：
    self._dataset
       ├── data_dict:      {文件名: DataFrame} — 实际的时序数据
       ├── covariate_dict: {文件名: 协变量信息} — 辅助信息
       └── metadata:       DataFrame — 每个文件的元信息索引
    """

    # 内部数据集类（可被子类替换）
    DATASET_CLASS = Dataset

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        初始化数据源
        
        参数：
            data_dict:      时序数据字典 {文件名: DataFrame}
            covariate_dict: 协变量字典 {文件名: 协变量信息}
            metadata:       元数据表（DataFrame），索引为文件名，列为属性字段
        """
        self._dataset = self.DATASET_CLASS()
        self._dataset.set_data(data_dict, covariate_dict, metadata)

    @property
    def dataset(self) -> Dataset:
        """返回内部维护的数据集对象"""
        return self._dataset

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        """
        按需加载一批时间序列
        
        这是抽象接口，需要子类实现具体的加载逻辑
        
        参数：series_list 待加载的文件名列表
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading series at runtime."
        )


class LocalDataSource(DataSource):
    """
    本地目录数据源：管理本地文件夹中的CSV数据文件
    
    目录结构约定：
    local_dataset_path/
       ├── data/              ← 存放所有CSV数据文件
       ├── covariates/        ← 存放协变量文件（可选）
       └── DETECT_META.csv    ← 元数据索引文件
    
    加载流程：
    1. 初始化时只加载元数据（轻量）
    2. 调用 load_series_list() 时才算真正加载数据（按需，节省内存）
    3. 使用线程池并行加载多个文件（加速）
    """

    # 元数据表的索引列名
    _INDEX_COL = "file_name"
    # 数据文件夹名称
    _DATA_FOLDER_NAME = "data"
    # 协变量文件夹名称
    _COVARIATES_FOLDER_NAME = "covariates"

    def __init__(self, local_dataset_path: str, metadata_file_name: str):
        """
        初始化本地数据源
        
        参数：
            local_dataset_path:  数据集的根目录路径
            metadata_file_name:  元数据文件名（如 "DETECT_META.csv"）
        """
        # 拼接完整路径
        self.local_data_path = os.path.join(local_dataset_path, self._DATA_FOLDER_NAME)
        self.local_covariates_path = os.path.join(
            local_dataset_path, self._COVARIATES_FOLDER_NAME
        )
        self.metadata_path = os.path.join(local_dataset_path, metadata_file_name)
        # 先更新元数据（检测是否有新文件加入），再加载
        metadata = self.update_meta_index()
        # 调用基类初始化，此时数据字典为空（之后按需加载）
        super().__init__({}, {}, metadata)

    def update_meta_index(self) -> pd.DataFrame:
        """
        检测并注册用户新增的数据文件
        
        功能：
        1. 从元数据CSV文件加载已有索引
        2. 扫描 data/ 文件夹，发现未注册的新CSV文件
        3. 自动提取新文件的信息并注册到元数据中
        4. 更新元数据CSV文件
        
        这是"即插即用"设计：用户只需把新CSV放到data/文件夹，
        系统会自动发现并注册。
        
        返回：更新后的元数据 DataFrame
        """
        metadata = self._load_metadata()
        # 扫描 data/ 目录中的所有 CSV 文件
        csv_files = {
            f
            for f in os.listdir(self.local_data_path)
            if f.endswith(".csv") and f != os.path.basename(self.metadata_path)
        }
        # 找出不在当前元数据中的新文件
        user_csv_files = set(csv_files).difference(metadata.index)
        if not user_csv_files:
            return metadata
        
        # 自动提取新文件的信息
        data_info_list = []
        for user_csv in user_csv_files:
            try:
                data_info_list.append(
                    load_series_info(os.path.join(self.local_data_path, user_csv))
                )
            except Exception as e:
                raise RuntimeError(f"Error loading series info from {user_csv}: {e}")
        
        # 合并到元数据并保存
        new_metadata = pd.DataFrame(data_info_list)
        new_metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        metadata = pd.concat([metadata, new_metadata])
        with open(self.metadata_path, "w", newline="", encoding="utf-8") as csvfile:
            metadata.to_csv(csvfile, index=False)
        logger.info(
            "Detected %s new user datasets, registered in the metadata",
            len(user_csv_files),
        )
        return metadata

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        """
        并行加载一批时间序列文件
        
        使用 ThreadPoolExecutor 多线程并发读取文件，
        显著加快大批量数据的加载速度。
        
        参数：series_list 待加载的文件名列表（不含路径，如 "MSL.csv"）
        """
        logger.info("Start loading %s series in parallel", len(series_list))
        
        # --- 并发加载时序数据 ---
        data_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_series, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            data_dict[series_name] = future.result()

        # --- 并发加载协变量数据 ---
        covariate_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_covariates, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            covariate_dict[series_name] = future.result()
        
        logger.info("Data loading finished.")
        # 更新到内部数据集中
        self.dataset.update_data(data_dict, covariate_dict)

    def _load_metadata(self) -> pd.DataFrame:
        """
        从本地CSV文件加载元数据表
        """
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def _load_series(self, series_name: str) -> pd.DataFrame:
        """
        加载单个时序文件
        
        参数：series_name 文件名（如 "MSL.csv"）
        返回：DataFrame 格式的时序数据
        """
        datafile_path = os.path.join(self.local_data_path, series_name)
        data = read_data(datafile_path)
        return data

    def _load_covariates(self, series_name: str) -> Optional[Dict]:
        """
        加载单个文件的协变量（辅助信息）
        
        协变量：可能是外部因素数据，如天气、节假日等
        
        参数：series_name 文件名
        """
        series_name_without_extension = os.path.splitext(series_name)[0]
        covariates_folder_path = os.path.join(
            self.local_covariates_path, series_name_without_extension
        )
        covariates = read_covariates(covariates_folder_path)
        return covariates


class LocalForecastingDataSource(LocalDataSource):
    """预测任务专用数据源（从 forecasting 目录加载）"""
    def __init__(self):
        super().__init__(FORECASTING_DATASET_PATH, "FORECAST_META.csv")


class LocalStForecastingDataSource(LocalDataSource):
    """时空预测任务专用数据源（从 st_forecasting 目录加载）"""
    def __init__(self):
        super().__init__(ST_FORECASTING_DATASET_PATH, "ST_FORECAST_META.csv")


class LocalAnomalyDetectDataSource(LocalDataSource):
    """
    异常检测任务专用数据源（★最常用）
    
    从 dataset/anomaly_detect/ 目录加载数据
    元数据文件为 DETECT_META.csv
    """
    def __init__(self):
        super().__init__(
            ANOMALY_DETECT_DATASET_PATH,
            "DETECT_META.csv",
        )
