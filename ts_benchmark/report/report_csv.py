# -*- coding: utf-8 -*-

import csv
import json
import os
from typing import Union, List

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.utils.leaderboard import get_leaderboard

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]


# TODO: update the docstring to match the common format in OTB.
def report(report_config: dict) -> None:
    """
    Generate a report based on specified configuration parameters.

    Parameters:
    - report_config (dict): A dictionary containing the following keys and their respective values:
        - log_files_list (List[str]): A list of file paths for log files.
        - leaderboard_file_name (str): The name for the saved report file.
        - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
        - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
        - fill_type (str): The type of fill for missing values.
        - null_value_threshold (float): The threshold value for null metrics.

    Raises:
    - ValueError: If all metrics have too many null values, making performance comparison impossible.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a CSV file.
    """
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if not log_files:
        raise ValueError("No log files to report")

    log_data = (
        log_files
        if isinstance(log_files, pd.DataFrame)
        else load_record_data(log_files, drop_columns=ARTIFACT_COLUMNS)
    )

    leaderboard_df = get_leaderboard(
        log_data,
        report_config["report_metrics"],
        report_config.get("aggregate_type", "mean"),
        report_config.get("fill_type", "mean_value"),
        report_config.get("null_value_threshold", 0.3),
    )

    num_rows = leaderboard_df.shape[0]
    strategy_arg_raw = log_data.iloc[0, 1]
    # 格式化JSON列：紧凑无空格，单行显示
    if isinstance(strategy_arg_raw, str):
        try:
            parsed = json.loads(strategy_arg_raw)
            strategy_arg_fmt = json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            strategy_arg_fmt = strategy_arg_raw
    else:
        strategy_arg_fmt = strategy_arg_raw
    leaderboard_df.insert(0, "strategy_args", [strategy_arg_fmt] * num_rows)
    
    # 格式化 model_params 列（如果存在）
    if "model_params" in leaderboard_df.columns:
        leaderboard_df["model_params"] = leaderboard_df["model_params"].apply(
            lambda x: json.dumps(x, separators=(',', ':'), ensure_ascii=False)
            if isinstance(x, dict) else (json.dumps(json.loads(x), separators=(',', ':'), ensure_ascii=False) if isinstance(x, str) and x.startswith("{") else x)
        )

    # 格式化耗时列为2位小数
    for col in ["fit_time", "inference_time"]:
        if col in leaderboard_df.columns:
            leaderboard_df[col] = leaderboard_df[col].apply(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else ""
            )

    # Create final DataFrame and save to CSV
    if report_config.get("save_path", None) is not None:
        save_path = report_config.get("save_path", None)
        leaderboard_df.to_csv(
            os.path.join(
                ROOT_PATH, "result", save_path, report_config["leaderboard_file_name"]
            ),
            index=False,
            float_format="%.6f",
        )
    else:
        leaderboard_df.to_csv(
            os.path.join(ROOT_PATH, "result", report_config["leaderboard_file_name"]),
            index=False,
            float_format="%.6f",
        )
