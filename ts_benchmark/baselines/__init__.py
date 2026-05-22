# -*- coding: utf-8 -*-
ADAPTER = {
    "darts_deep_model_adapter": "ts_benchmark.baselines.darts.darts_deep_model_adapter",
    "darts_statistical_model_adapter": "ts_benchmark.baselines.darts.darts_statistical_model_adapter",
    "darts_regression_model_adapter": "ts_benchmark.baselines.darts.darts_regression_model_adapter",
    "transformer_adapter": "ts_benchmark.baselines.time_series_library.adapters_for_transformers.transformer_adapter",
}

# === 短名称导出 ===
# 让 --model-name LaGraph 这样的短名称也能解析
LaGraph = "ts_benchmark.baselines.self_impl.LaGraph.LaGraph.LaGraph"
