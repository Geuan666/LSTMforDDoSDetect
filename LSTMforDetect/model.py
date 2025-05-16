#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """用于序列模型的自注意力机制"""

    def __init__(self, hidden_size):
        """
        初始化自注意力模块
        参数:
            hidden_size: 隐藏状态的维度
        """
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        # 查询、键、值投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 点积注意力的缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

        # 输出投影
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        """
        自注意力的前向传播
        参数:
            hidden_states: 隐藏状态序列 [batch_size, seq_len, hidden_size]
        返回:
            context: 注意力加权的上下文向量 [batch_size, seq_len, hidden_size]
            attention_weights: 用于可视化的注意力权重 [batch_size, seq_len, seq_len]
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # 创建查询、键、值投影
        query = self.query(hidden_states)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_states)  # [batch_size, seq_len, hidden_size]
        value = self.value(hidden_states)  # [batch_size, seq_len, hidden_size]

        # 缩放点积注意力
        # 转置键用于点积: [batch_size, hidden_size, seq_len]
        key_t = key.permute(0, 2, 1)

        # 注意力分数: [batch_size, seq_len, seq_len]
        energy = torch.bmm(query, key_t) / self.scale.to(hidden_states.device)

        # 将注意力分数归一化为概率
        attention_weights = F.softmax(energy, dim=2)

        # 将注意力权重应用于值
        # [batch_size, seq_len, seq_len] x [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        context = torch.bmm(attention_weights, value)

        # 应用输出投影
        context = self.fc_out(context)

        return context, attention_weights


class DDoSDetector(nn.Module):
    """
    使用LSTM和自注意力的DDoS检测模型
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=8, dropout_rate=0.3):
        """
        初始化DDoS检测模型
        参数:
            input_size: 每个时间步的输入特征数量
            hidden_size: LSTM隐藏状态维度
            num_layers: LSTM层数
            num_classes: 输出类别数量
            dropout_rate: Dropout概率
        """
        super(DDoSDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # LSTM层 (单向)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False  # 单向
        )

        # 自注意力机制
        self.attention = SelfAttention(hidden_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 用于分类的全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 因为我们将连接最后的隐藏状态和注意力

        # 初始化权重
        self._init_weights()

        logger.info(f"初始化DDoSDetector: input_size={input_size}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}, "
                    f"num_classes={num_classes}, dropout_rate={dropout_rate}")

    def _init_weights(self):
        """初始化模型权重以获得更好的收敛性"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重: 使用正交初始化
                    nn.init.orthogonal_(param)
                else:
                    # 线性层: 使用xavier初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 将偏置初始化为零
                nn.init.zeros_(param)

    def forward(self, x, return_attention=False):
        """
        模型的前向传播
        参数:
            x: 输入张量 [batch_size, seq_len, input_size]
            return_attention: 是否返回注意力权重用于可视化
        返回:
            output: 类别逻辑值 [batch_size, num_classes]
            attention_weights: 可选，注意力权重 [batch_size, seq_len, seq_len]
        """
        # 检查输入形状
        batch_size, seq_len, input_size = x.shape

        # LSTM前向传播
        lstm_out, (final_hidden, _) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size]
        # final_hidden: [num_layers, batch_size, hidden_size]

        # 获取最终层的所有时间步的隐藏状态
        if self.num_layers > 1:
            lstm_out = lstm_out.view(batch_size, seq_len, self.hidden_size)

        # 提取最后一个时间步的隐藏状态
        last_hidden = final_hidden[-1]  # [batch_size, hidden_size]

        # 应用自注意力
        context, attention_weights = self.attention(lstm_out)

        # 使用注意力加权和作为上下文向量
        # 在序列维度上求和得到 [batch_size, hidden_size]
        context_vector = torch.sum(context, dim=1)

        # 将最后的隐藏状态与上下文向量连接
        combined = torch.cat((last_hidden, context_vector), dim=1)  # [batch_size, hidden_size*2]

        # 应用dropout进行正则化
        combined = self.dropout(combined)

        # 通过全连接层
        output = self.fc(combined)  # [batch_size, num_classes]

        if return_attention:
            return output, attention_weights
        return output