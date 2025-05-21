#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class BiLSTMDetector(nn.Module):
    """
    用于DDoS攻击检测的简化LSTM模型
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=13, dropout_rate=0.3):
        """
        初始化DDoS检测模型
        参数:
            input_size: 每个时间步的输入特征数量
            hidden_size: LSTM隐藏状态维度
            num_layers: LSTM层数
            num_classes: 输出类别数量
            dropout_rate: Dropout概率
        """
        super(BiLSTMDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # 双向LSTM层 - 能够同时捕获前向和后向的时间依赖性
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 批归一化层 - 加速收敛并提高泛化能力
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # 分类器 - 使用两层全连接网络
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # 初始化权重
        self._init_weights()

        logger.info(f"初始化SimpleLSTMDetector: input_size={input_size}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}, "
                    f"num_classes={num_classes}, dropout_rate={dropout_rate}")

    def _init_weights(self):
        """初始化模型权重，针对不同类型的参数使用合适的初始化方法"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重: 使用正交初始化
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:  # 只对2维及以上的张量使用xavier初始化
                    # 线性层权重: 使用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 1维权重(如batch_norm的weight): 使用常数1初始化
                    nn.init.constant_(param, 0.0894)
            elif 'bias' in name:
                # 将偏置初始化为零
                nn.init.zeros_(param)

    def forward(self, x):
        """
        模型的前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, input_size]
        返回:
            output: 类别逻辑值 [batch_size, num_classes]
        """
        # 检查输入形状
        batch_size, seq_len, _ = x.shape

        # LSTM前向传播
        lstm_out, (final_hidden, _) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size*2] (因为是双向LSTM)

        # 获取最后一层、最后一个时间步的隐藏状态
        # 对于双向LSTM，需要连接两个方向的最终隐藏状态
        # final_hidden的形状: [num_layers*2, batch_size, hidden_size]

        # 重新整形最终隐藏状态以提取最后一层
        final_hidden = final_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        final_forward = final_hidden[-1, 0, :, :]  # 最后一层前向LSTM的隐藏状态
        final_backward = final_hidden[-1, 1, :, :]  # 最后一层后向LSTM的隐藏状态

        # 连接前向和后向状态
        combined = torch.cat((final_forward, final_backward), dim=1)  # [batch_size, hidden_size*2]

        # 应用批归一化
        combined = self.batch_norm(combined)

        # 应用dropout进行正则化
        combined = self.dropout(combined)

        # 通过第一个全连接层
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)

        # 通过第二个全连接层得到最终输出
        output = self.fc2(x)  # [batch_size, num_classes]

        return output