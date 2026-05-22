# -*- coding: utf-8 -*-
"""
DCdetector — 基于双重视角对比的异常检测模型
============================================
DC = Dual Contrastive（双重对比）

核心思想：
  - 从多个patch（子序列）视角看同一个时间窗口
  - 不同视角之间应该有相似的分布（正常模式）
  - 如果后验和先验的分布差异大 → 说明有异常

工作流程：
  训练：
    1. 标准化数据 → 创建滑动窗口
    2. 模型将窗口分成多个patch，从每个patch提取特征
    3. 计算后验分布（模型认为的实际分布）
    4. 计算先验分布（基于时间距离的期望分布）
    5. 用KL散度衡量两者差异，最大化差异来学习区分特征
    6. 早停保存最佳模型
    
  推理：
    1. 用KL散度计算每个窗口的异常分数
    2. 用训练集的分数分布确定阈值
    3. 超过阈值的标记为异常

与LaGraph的区别：
  - LaGraph: 用图结构学习通道间关系，重建正常模式
  - DCdetector: 用多patch对比学习，对比视角一致性

类比理解：
  就像从不同角度（正面/侧面）看同一个物体，正常情况下看到的应该是一致的。
  如果某个角度看出了异常的模式（分布差异大），那这个物体（时间窗口）就有问题。
"""

import copy
import time
from typing import Type, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.self_impl.DCdetector.DCdetector_model import DCdetector_model
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split

# ============================================================
# 默认超参数配置
# ============================================================
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,            # 窗口大小：每次看100个时间步
    "patch_size": [5],          # patch大小列表：[5]表示每个patch有5个时间步
                                # 多个值则可以同时从不同粒度看数据
    "lr": 0.0001,               # 学习率
    "n_heads": 1,               # 注意力头数
    "e_layers": 3,              # 编码器层数
    "d_model": 256,             # 模型隐藏维度
    "rec_timeseries": True,     # 是否重建时间序列
    "num_epochs": 10,           # 训练轮数
    "batch_size": 128,          # 批次大小
    "patience": 5,              # 早停耐心值：连续5轮不提升就停止
    "k": 3,                     # top-k最近邻（用于图构建）
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
                                # 异常比例列表
}


def my_kl_loss(p, q):
    """
    计算KL散度（Kullback-Leibler Divergence）损失
    
    KL散度衡量两个概率分布的差异：
    - 如果两个分布完全相同，KL散度 = 0
    - 差异越大，KL散度越大
    
    公式: KL(p||q) = p * (log(p) - log(q))
    
    DCdetector用KL散度来比较：
    - p: 模型学习到的后验分布（实际）
    - q: 基于先验的期望分布
    差异越大 → 越可能是异常
    
    参数：
        p: 后验分布
        q: 先验分布
    返回：
        所有head的平均KL散度
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    """
    学习率衰减策略
    
    每过一个epoch，学习率乘以0.5
    这有助于：前期大步快走，后期小步精调
    
    参数：
        optimizer: 优化器
        epoch: 当前轮数
        lr_: 初始学习率
    """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStopping:
    """
    早停机制：防止过拟合
    
    原理：
    - 训练时监控验证集的损失
    - 如果连续 patience 轮验证损失都没有下降
    - 就提前停止训练，避免在训练集上过拟合
    
    与CATCH的EarlyStopping不同：
    - DCdetector 同时看两个损失（val_loss1 和 val_loss2）
    - 只要有一个没有改善就计数
    
    参数：
        patience: 耐心值（容忍多少轮不提升）
        verbose: 是否打印详细信息
        delta: 最小改进阈值（小于这个数值不算提升）
    """
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience       # 容忍轮数
        self.verbose = verbose         # 是否打印
        self.counter = 0               # 已连续不提升的轮数
        self.best_score = None         # 历史最佳分数
        self.best_score2 = None        # 第二个损失的历史最佳
        self.early_stop = False        # 是否触发早停
        self.val_loss_min = np.Inf     # 最小验证损失（初始化为无穷大）
        self.val_loss2_min = np.Inf    # 第二个损失的最小值
        self.delta = delta             # 改进阈值
        self.dataset = dataset_name
        self.check_point = None        # 最佳模型参数（用于后续加载）

    def __call__(self, val_loss, val_loss2, model):
        """
        每轮训练结束后调用，判断是否应该早停
        
        参数：
            val_loss: 验证集损失1
            val_loss2: 验证集损失2
            model: 当前模型（用于保存最佳参数）
        """
        score = -val_loss       # 分数 = 负损失（越大越好）
        score2 = -val_loss2
        
        if self.best_score is None:
            # 第一次：初始化最佳分数
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
        elif (
            score < self.best_score + self.delta      # 没有提升
            or score2 < self.best_score2 + self.delta  # 第二个也没提升
        ):
            # 任何一个损失没有提升 → 计数+1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停！
        else:
            # 有提升 → 重置计数，保存最佳模型
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model):
        """
        保存当前最佳模型参数
        
        用 deepcopy 复制参数，避免后续训练覆盖最佳状态。
        """
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        self.check_point = copy.deepcopy(model.state_dict())


class TransformerConfig:
    """
    配置类：统一管理DCdetector模型的所有超参数
    
    用法同CATCH的TransformerConfig：
        config = TransformerConfig(win_size=200, lr=0.001)
        # 未指定的参数使用默认值
    """
    def __init__(self, **kwargs):
        # 先设置所有默认值
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        # 再用用户传入的参数覆盖
        for key, value in kwargs.items():
            setattr(self, key, value)


class DCdetector:
    """
    DCdetector 异常检测模型主类
    
    接口与CATCH一致：
    - detect_fit(train, test): 训练模型
    - detect_score(test): 返回异常分数
    - detect_label(test): 返回异常标签
    
    核心区别：
    - 训练目标：最大化先验和后验之间的KL散度（对比学习）
    - 异常分数：后验对先验的偏离程度
    
    用法示例：
        model = DCdetector(win_size=100, num_epochs=10)
        model.detect_fit(train_df, test_df)
        scores = model.detect_score(test_df)
        labels = model.detect_label(test_df)
    """
    
    def __init__(self, **kwargs):
        super(DCdetector, self).__init__()
        self.config = TransformerConfig(**kwargs)
        # 标准化器
        self.scaler = StandardScaler()
        self.win_size = self.config.win_size
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def required_hyper_params() -> dict:
        """返回模型所需的超参数（框架接口）"""
        return {}

    def __repr__(self) -> str:
        return self.model_name

    def vali(self, vali_loader):
        """
        在验证集上评估模型性能
        
        DCdetector的验证指标：
        - 计算先验损失（prior_loss）和后验损失（series_loss）
        - prior_loss: 先验分布对后验的KL散度
        - series_loss: 后验分布对先验的KL散度
        - 用两者的差值作为验证指标
        
        KL散度的计算使用了对称KL：
        KL_对称(p, q) = KL(p||q) + KL(q||p)
        这样可以衡量两个分布的双向差异
        
        参数：
            vali_loader: 验证数据加载器
        返回：
            (avg_loss1, avg_loss2): 平均损失
        """
        self.model.eval()
        loss_1 = []
        loss_2 = []
        
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            # 模型前向传播：返回后验(series)和先验(prior)
            series, prior = self.model(input)
            
            series_loss = 0.0  # 后验损失
            prior_loss = 0.0   # 先验损失
            
            for u in range(len(prior)):
                # 对先验分布做归一化（归一化到概率分布，每行和为1）
                # prior[u] / sum(prior[u], dim=-1) → 得到概率分布
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                # 对称KL散度：series_loss + prior_loss
                # KL(p||q) + KL(q||p)
                series_loss += torch.mean(
                    my_kl_loss(series[u], prior_normalized.detach())
                ) + torch.mean(
                    my_kl_loss(prior_normalized.detach(), series[u])
                )
                
                prior_loss += torch.mean(
                    my_kl_loss(prior_normalized, series[u].detach())
                ) + torch.mean(
                    my_kl_loss(series[u].detach(), prior_normalized)
                )

            # 取所有层的平均
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            # 损失 = 先验损失 - 后验损失
            # 这个差值越大越好（说明后验分布更偏离先验）
            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型
        
        训练流程：
        1. 设置输入输出维度
        2. 划分训练/验证集 (8:2)
        3. 标准化数据
        4. 创建数据加载器
        5. 构建模型
        6. 循环训练 → 验证 → 早停
        
        训练目标：
        - 对所有层，最小化后验与先验的KL散度
        - 这反向推动了模型学习更好的特征表示
        
        参数：
            train_data: 训练数据
            test_data: 测试数据（用于维度设置）
        """
        # --- 步骤1: 设置维度 ---
        self.config.input_c = train_data.shape[1]   # 输入变量数
        self.config.output_c = train_data.shape[1]  # 输出变量数

        # --- 步骤2: 划分数据集 (80%训练, 20%验证) ---
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        
        # --- 步骤3: 标准化 ---
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

        # --- 步骤4: 创建数据加载器 ---
        self.train_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="train",
        )
        self.valid_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="val",
        )

        # --- 步骤5: 构建模型 ---
        self.model = DCdetector_model(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
        )
        self.model.to(self.device)
        
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        # --- 步骤6: 初始化早停和优化器 ---
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        time_now = time.time()
        train_steps = len(self.train_loader)

        # --- 步骤7: 训练循环 ---
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                
                # 前向传播：获取后验和先验
                series, prior = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0

                # 对每一层计算对称KL散度
                for u in range(len(prior)):
                    prior_normalized = (
                        prior[u]
                        / torch.unsqueeze(
                            torch.sum(prior[u], dim=-1), dim=-1
                        ).repeat(1, 1, 1, self.config.win_size)
                    )
                    
                    # 对称KL: KL(series||prior) + KL(prior||series)
                    series_loss += torch.mean(
                        my_kl_loss(series[u], prior_normalized.detach())
                    ) + torch.mean(
                        my_kl_loss(prior_normalized.detach(), series[u])
                    )
                    
                    prior_loss += torch.mean(
                        my_kl_loss(prior_normalized, series[u].detach())
                    ) + torch.mean(
                        my_kl_loss(series[u].detach(), prior_normalized)
                    )

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss

                # 每100步打印进度
                if (i + 1) % 100 == 0:
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

            # 验证 + 早停
            vali_loss1, vali_loss2 = self.vali(self.valid_loader)
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            self.early_stopping(vali_loss1, vali_loss2, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 调整学习率
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型计算异常分数
        
        异常分数计算：
        1. 加载最佳模型
        2. 对每个窗口，计算所有层的后验 vs 先验KL散度
        3. 用softmax将各层分数融合
        4. temperature=50：放大分数差异（温度越高，差异越明显）
        
        温度参数的作用：
        - temperature 越大，softmax后的分数差异越大
        - 有助于更好地区分正常和异常
        
        参数：
            train: 测试数据 DataFrame
        返回：
            (test_energy, test_energy): 异常分数数组
        """
        # 加载最佳模型
        self.model.load_state_dict(self.early_stopping.check_point)

        # 标准化
        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        self.model.eval()
        temperature = 50  # 温度参数：放大异常分数差异

        test_labels = []
        attens_energy = []
        
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    prior_normalized = (
                        prior[u]
                        / torch.unsqueeze(
                            torch.sum(prior[u], dim=-1), dim=-1
                        ).repeat(1, 1, 1, self.config.win_size)
                    )
                    
                    if u == 0:
                        # 第一层：初始化
                        series_loss = (
                            my_kl_loss(series[u], prior_normalized.detach())
                            * temperature
                        )
                        prior_loss = (
                            my_kl_loss(prior_normalized, series[u].detach())
                            * temperature
                        )
                    else:
                        # 后续层：累加
                        series_loss += (
                            my_kl_loss(series[u], prior_normalized.detach())
                            * temperature
                        )
                        prior_loss += (
                            my_kl_loss(prior_normalized, series[u].detach())
                            * temperature
                        )
                
                # softmax融合所有层的分数
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型预测异常标签
        
        工作流程：
        1. 计算训练集的异常分数分布
        2. 计算测试集的异常分数
        3. 合并两部分的分数
        4. 用不同的 anomaly_ratio 确定阈值
        5. 超过阈值的标记为异常（1）
        
        参数：
            train: 测试数据
        返回：
            (preds, test_energy):
                - preds: {anomaly_ratio: 预测标签数组}
                - test_energy: 异常分数数组
        """
        self.model.load_state_dict(self.early_stopping.check_point)

        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        self.model.eval()
        temperature = 50

        # --- 步骤1: 计算训练集异常分数 ---
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # --- 步骤2: 计算测试集异常分数 ---
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        # --- 步骤3: 合并分数 ---
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # --- 步骤4: 为每个 anomaly_ratio 确定阈值 ---
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # --- 步骤5: 生成预测标签 ---
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            # 取分位数作为阈值
            # 例如 ratio=1.0: 取第99百分位 → 只有1%的会被判为异常
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
            
        return preds, test_energy
