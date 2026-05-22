# -*- coding: utf-8 -*-
"""
CATCH — 基于频域+时域联合的异常检测模型
========================================
CATCH = Channel ATtention for time series anomaly detection（通道注意力异常检测）

核心思想：
  - 同时从时域和频域两个角度检测异常
  - 时域：用重建误差（MSE）判断序列是否"正常"
  - 频域：用频率损失判断频谱模式是否异常
  - 通道发现：自动学习哪些变量之间有联系

工作流程：
  训练阶段：
    1. 标准化数据
    2. 创建滑动窗口样本
    3. 模型重建 + 频率重建
    4. 计算 重建损失 + 频率损失 + 通道发现损失
    5. 反向传播更新参数
    6. 早停机制防止过拟合
  
  推理阶段：
    1. 对测试数据计算重建误差
    2. 用训练数据的误差分布确定阈值
    3. 超过阈值的标记为异常

与LaGraph的关系：
  - 两者都用于多变量时间序列异常检测
  - LaGraph 用图卷积 + 注意力，侧重通道间结构关系
  - CATCH 用通道掩码 + 频率损失，侧重频域信息

类比理解：
  就像同时用眼睛（时域重建）和耳朵（频域分析）来判断机器是否正常运转。
  眼睛看数值是否偏离，耳朵听声音频谱是否异常。
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.catch.models.CATCH_model import (
    CATCHModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate

# ============================================================
# 默认超参数配置
# ============================================================
# 这些是 CATCH 模型的默认设置，用户可以通过 kwargs 覆盖任意参数
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,              # 学习率：每次更新参数时调整的步长
    "Mlr": 0.00001,            # 掩码生成器的学习率（独立优化，通常比主模型小）
    "e_layers": 3,             # 编码器层数：堆叠几层Transformer
    "n_heads": 2,              # 多头注意力头数
    "cf_dim": 64,              # 通道特征维度
    "d_ff": 256,               # 前馈网络中间层维度
    "d_model": 128,            # 模型隐藏层维度（主干网络的宽度）
    "head_dim": 64,            # 每个注意力头的维度
    "individual": 0,           # 是否每个通道独立预测（0=共享，1=独立）
    "dropout": 0.2,            # Dropout比例：训练时随机丢弃20%的神经元，防过拟合
    "head_dropout": 0.1,       # 注意力头的Dropout比例
    "auxi_loss": "MAE",        # 辅助损失函数类型（MAE=平均绝对误差）
    "auxi_type": "complex",    # 辅助损失类型（complex=复频域）
    "auxi_mode": "fft",        # 频率分析模式（fft=快速傅里叶变换）
    "auxi_lambda": 0.005,      # 辅助损失权重：频率损失在总损失中的占比
    "score_lambda": 0.05,      # 异常分数中频率分数的权重
    "regular_lambda": 0.5,     # 正则化权重
    "temperature": 0.07,       # 温度参数（用于对比学习）
    "patch_stride": 8,         # 训练时 patch 的滑动步长
    "patch_size": 16,          # 训练时 patch 的大小（每16个时间步为一块）
    "inference_patch_stride": 1,  # 推理时 patch 的滑动步长（更精细）
    "inference_patch_size": 32,   # 推理时 patch 的大小
    "dc_lambda": 0.005,        # 通道发现损失权重
    "module_first": True,      # 是否先通过模块再归一化
    "mask": False,             # 是否使用掩码
    "pretrained_model": None,  # 预训练模型路径
    "num_epochs": 3,           # 训练轮数
    "batch_size": 128,         # 批次大小：每次处理128个窗口样本
    "patience": 3,             # 早停耐心值：连续3轮验证集不提升就停止
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
                               # 异常比例列表：用于确定异常阈值
                               # 例如0.1表示假设数据中0.1%是异常
    "seq_len": 192,            # 序列长度（窗口大小）：每次看192个时间步
    "pct_start": 0.3,          # OneCycleLR中学习率上升阶段占比
    "revin": 1,                # 是否使用RevIN可逆归一化（1=使用，0=不使用）
    "affine": 0,               # RevIN的仿射变换（1=学习缩放和平移）
    "subtract_last": 0,        # 是否减去最后一个值
    "lradj": "type1",          # 学习率调整策略
}


class TransformerConfig:
    """
    配置类：统一管理CATCH模型的所有超参数
    
    用法：
        config = TransformerConfig(lr=0.001, seq_len=256)
        # 未指定的参数使用默认值
        print(config.lr)        # 0.001
        print(config.seq_len)   # 256
        print(config.d_model)   # 128（默认值）
    
    属性访问：
        pred_len 和 learning_rate 是计算属性（property），
        方便与其他模型接口兼容。
    """
    def __init__(self, **kwargs):
        # 先设置所有默认值
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        # 再用用户传入的参数覆盖默认值
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        """预测长度：等于序列长度（做重建任务）"""
        return self.seq_len

    @property
    def learning_rate(self):
        """学习率的别名（兼容不同命名习惯）"""
        return self.lr


class CATCH:
    """
    CATCH 异常检测模型主类
    
    这是给外部调用的接口类，内部包装了 CATCHModel。
    提供标准的 detect_fit/ detect_score/ detect_label 接口。
    
    训练方法：detect_fit(train_data, test_data)
    - 标准化 → 创建窗口 → 模型训练 → 早停保存最佳模型
    
    推理方法：
    - detect_score(test): 返回异常分数（连续值，越大越异常）
    - detect_label(test): 返回异常标签（0/1二值，超过阈值为1）
    
    使用方法：
        model = CATCH(seq_len=192)
        model.detect_fit(train_df, test_df)
        scores = model.detect_score(test_df)
        labels = model.detect_label(test_df)
    """
    
    def __init__(self, **kwargs):
        super(CATCH, self).__init__()
        # 创建配置对象
        self.config = TransformerConfig(**kwargs)
        # 标准化器：将数据缩放到均值0、方差1
        self.scaler = StandardScaler()
        # 设备选择：有GPU就用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 主损失函数：均方误差（MSE）
        self.criterion = nn.MSELoss()
        # 频率损失函数：对比原始和重建的频谱
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len

    @staticmethod
    def required_hyper_params() -> dict:
        """
        返回模型所需的超参数（给框架调用的接口）
        
        :return: 空字典，表示模型不需要额外超参数
        """
        return {}

    def __repr__(self) -> str:
        """返回模型名称的字符串表示"""
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        """
        根据训练数据自动调整超参数
        
        主要工作：
        1. 推断数据的时间频率（秒/分/时/天）
        2. 根据数据的列数设置输入输出维度
        
        参数：
            train_data: 训练数据 DataFrame
        """
        # --- 推断时间频率 ---
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'  # 默认为秒
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        # --- 根据数据维度设置模型参数 ---
        column_num = train_data.shape[1]  # 变量数（通道数）
        self.config.enc_in = column_num    # 编码器输入维度
        self.config.dec_in = column_num    # 解码器输入维度
        self.config.c_out = column_num     # 输出维度
        self.config.label_len = 48         # 标签长度（用于预测任务）

    def detect_validate(self, valid_data_loader, criterion):
        """
        在验证集上评估模型性能
        
        返回验证损失（越小越好），用于早停判断。
        
        参数：
            valid_data_loader: 验证数据加载器
            criterion: 损失函数
        返回：
            平均验证损失
        """
        config = self.config
        total_loss = []
        self.model.eval()  # 切换到评估模式（关闭Dropout等）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():  # 不计算梯度，节省内存和计算
            for input, _ in valid_data_loader:
                input = input.to(device)
                # 模型前向传播
                output, _, _ = self.model(input)
                output = output[:, :, :]
                # 计算损失
                output = output.detach().cpu()
                true = input.detach().cpu()
                loss = criterion(output, true).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)  # 平均所有批次的损失
        self.model.train()  # 切换回训练模式
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型（核心训练循环）
        
        训练流程：
        1. 自动调整超参数（根据数据维度）
        2. 划分训练集和验证集（8:2）
        3. 标准化数据
        4. 创建滑动窗口数据加载器
        5. 循环训练 → 验证 → 早停
        6. 保存最佳模型
        
        损失组成：
        - 重建损失（rec_loss）：MSE(input, output)
        - 频率损失（auxi_loss）：频谱差异
        - 通道发现损失（dcloss）：变量间关系学习
        总损失 = rec_loss + dc_lambda*dcloss + auxi_lambda*auxi_loss
        
        参数：
            train_data: 训练数据
            test_data:  测试数据（用于设置维度，不参与训练）
        """
        # --- 步骤1: 自动调整超参数 ---
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        
        # --- 步骤2: 创建模型 ---
        self.model = CATCHModel(self.config)
        self.model.to(self.device)
        config = self.config
        
        # --- 步骤3: 划分训练/验证集（80%训练，20%验证）---
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        
        # --- 步骤4: 标准化 ---
        self.scaler.fit(train_data_value.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        # --- 步骤5: 创建数据加载器 ---
        # 验证数据加载器
        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )
        # 训练数据加载器
        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        # 打印可训练参数数量
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        # --- 步骤6: 初始化早停 ---
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        
        # --- 步骤7: 分离参数组 ---
        # 主模型参数（排除 mask_generator）
        main_params = [param for name, param in self.model.named_parameters() 
                      if 'mask_generator' not in name]
        # 掩码生成器单独优化（使用不同的学习率）
        self.optimizer = torch.optim.Adam(main_params, lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(
            self.model.mask_generator.parameters(), lr=self.config.Mlr
        )

        # --- 步骤8: 学习率调度器 ---
        # OneCycleLR：学习率先升后降，帮助更快收敛
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,  # 前30%的时间学习率上升
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )

        time_now = time.time()

        # --- 步骤9: 训练循环 ---
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()
            
            step = min(int(len(self.train_data_loader) / 10), 100)
            
            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)

                # 前向传播
                output, output_complex, dcloss = self.model(input)
                output = output[:, :, :]

                # 计算三类损失
                rec_loss = self.criterion(output, input)  # 重建损失
                norm_input = self.model.revin_layer(input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)  # 频率损失
                # 总损失 = 重建 + 通道发现 + 频率
                loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * auxi_loss

                train_loss.append(loss.item())

                # 每隔一定步数更新掩码生成器
                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()

                # 每100步打印进度
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} "
                        "| training fre loss: {3:.7f} | training dc loss: {4:.7f}".format(
                            i + 1, epoch + 1, rec_loss.item(), auxi_loss.item(), dcloss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                            (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # 反向传播
                loss.backward()
                self.optimizer.step()

            # --- 每个epoch结束后的验证 ---
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )

            # 早停检查
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率
            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型计算异常分数
        
        异常分数的计算：
        1. 对每个窗口，用模型做重建
        2. 计算时域分数：MSE(原始, 重建)
        3. 计算频域分数：频谱差异
        4. 最终分数 = 时域分数 + score_lambda * 频域分数
        5. 分数越大 → 越异常
        
        参数：
            test: 测试数据 DataFrame
        返回：
            (test_energy, test_energy): 异常分数数组（两个返回值相同）
        """
        # 标准化
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        # 加载训练时保存的最佳模型
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        # 创建数据加载器
        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        # 不取平均的MSE（保留每个时间步的重建误差）
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        
        attens_energy = []
        test_labels = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # 模型重建
                outputs, _, _ = self.model(batch_x)
                # 时域分数：每个通道的重建误差取平均
                temp_score = torch.mean(
                    self.temp_anomaly_criterion(batch_x, outputs), dim=-1
                )
                # 频域分数
                freq_score = torch.mean(
                    self.freq_anomaly_criterion(batch_x, outputs), dim=-1
                )
                # 最终分数
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型预测异常标签（0=正常, 1=异常）
        
        工作流程：
        1. 分别计算训练集和测试集的异常分数
        2. 用不同的 anomaly_ratio 确定多个阈值
        3. 超过阈值的标记为异常
        4. 返回多个阈值下的预测结果
        
        为什么使用多个 anomaly_ratio？
        - 不同场景下异常比例不同
        - 提供多个阈值让用户选择最合适的
        
        参数：
            test: 测试数据 DataFrame
        返回：
            (preds, test_energy): 
                - preds: 字典，key=anomaly_ratio，value=预测标签数组
                - test_energy: 异常分数数组
        """
        # 标准化测试数据
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        # 加载最佳模型
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        # --- 创建各种数据加载器 ---
        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )
        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []
        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)

        # --- 步骤1: 计算训练集的异常分数（用于确定阈值）---
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x)
                temp_score = torch.mean(
                    self.temp_anomaly_criterion(batch_x, outputs), dim=-1
                )
                freq_score = torch.mean(
                    self.freq_anomaly_criterion(batch_x, outputs), dim=-1
                )
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # --- 步骤2: 计算测试集的异常分数 ---
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x)
                temp_score = torch.mean(
                    self.temp_anomaly_criterion(batch_x, outputs), dim=-1
                )
                freq_score = torch.mean(
                    self.freq_anomaly_criterion(batch_x, outputs), dim=-1
                )
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        # 合并训练和测试分数
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # --- 步骤3: 用多个比率确定阈值 ---
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x)
                temp_score = torch.mean(
                    self.temp_anomaly_criterion(batch_x, outputs), dim=-1
                )
                freq_score = torch.mean(
                    self.freq_anomaly_criterion(batch_x, outputs), dim=-1
                )
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # 确保 anomaly_ratio 是列表
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        # --- 步骤4: 为每个 ratio 生成标签 ---
        # np.percentile: 计算分位数作为阈值
        # 例如 ratio=1.0: 取 combined_energy 的第99百分位
        # 即假设只有1%的数据是异常的
        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
