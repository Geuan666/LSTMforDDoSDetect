# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.svm import SVC
import os
import pickle

logger = logging.getLogger(__name__)


class BiLSTMDetector(nn.Module):
    """
    用于DDoS攻击检测的双向LSTM模型
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

        logger.info(f"初始化BiLSTMDetector: input_size={input_size}, "
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
                    # 1维权重(如batch_norm的weight): 使用常数初始化
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


class SVMModel:
    """SVM分类器模型类"""

    def __init__(self, class1, class2, kernel='rbf', C=1.0, gamma='scale', probability=True):
        """
        初始化SVM模型

        参数:
            class1: 第一个类别的索引
            class2: 第二个类别的索引
            kernel: 核函数类型
            C: 正则化参数
            gamma: 核系数
            probability: 是否启用概率估计
        """
        self.class1 = class1
        self.class2 = class2
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            class_weight='balanced'
        )

    def fit(self, X, y):
        """
        训练SVM模型

        参数:
            X: 特征矩阵
            y: 标签向量（二进制，0表示class1，1表示class2）
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        预测样本类别

        参数:
            X: 特征矩阵

        返回:
            预测的类别: 返回原始类别索引（class1或class2）
        """
        binary_pred = self.model.predict(X)
        # 将二元预测转换回原始类别
        return np.where(binary_pred == 0, self.class1, self.class2)

    def save(self, path):
        """
        保存模型

        参数:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        加载模型

        参数:
            path: 模型路径

        返回:
            加载的SVM模型实例
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


class SVMCascadeModel:
    """
    SVM级联模型：结合BiLSTM基础分类器和SVM二分类器
    """

    def __init__(self, base_model, confusion_pairs=None, confidence_threshold=0.95):
        """
        初始化级联模型
        参数:
            base_model: 基础BiLSTM模型
            confusion_pairs: 混淆类别对列表，如 [(11,12), (5,7), (2,8), (9,10)]
            confidence_threshold: 不触发二级分类器的置信度阈值
        """
        self.base_model = base_model
        self.confusion_pairs = confusion_pairs or [(11, 12), (5, 7), (2, 8), (9, 10)]
        self.confidence_threshold = confidence_threshold
        self.svm_models = {}

    def load_svm_models(self, model_dir):
        """
        加载SVM二分类器
        参数:
            model_dir: SVM模型目录
        """
        for class1, class2 in self.confusion_pairs:
            model_path = os.path.join(model_dir, f"svm_model_{class1}_{class2}.pkl")
            if os.path.exists(model_path):
                self.svm_models[(class1, class2)] = SVMModel.load(model_path)
                logger.info(f"已加载SVM分类器: {class1} vs {class2}")
            else:
                logger.warning(f"找不到SVM分类器: {class1} vs {class2}")

    def predict(self, inputs, device=None):
        """
        使用级联模型预测

        参数:
            inputs: BiLSTM模型的输入数据，形状为[batch_size, feature_dim, 1]
            device: 计算设备

        返回:
            final_pred: 最终预测标签
            base_pred: 基础模型预测
            probs: 预测概率
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基础BiLSTM模型预测
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(inputs.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            base_pred = outputs.max(1)[1].cpu().numpy()
            confidence = probs.max(1)[0].cpu().numpy()

        # 如果没有加载SVM分类器，直接返回基础预测
        if not self.svm_models:
            return base_pred, base_pred, probs.cpu().numpy()

        # 最终预测结果
        final_pred = base_pred.copy()

        # 提取PCA降维后的特征（从输入的倒数第二维）
        pca_features = inputs.squeeze(-1).cpu().numpy()

        # 对每个样本应用SVM二次分类
        for i, (pred, conf) in enumerate(zip(base_pred, confidence)):
            # 检查是否是混淆类别对
            for class1, class2 in self.confusion_pairs:
                if pred in [class1, class2] and conf < self.confidence_threshold:
                    # 找到对应的SVM分类器
                    svm_model = self.svm_models.get((class1, class2))
                    if svm_model is None:
                        continue

                    # 获取当前样本的特征
                    feature = pca_features[i:i + 1]

                    # 预测并更新结果
                    final_pred[i] = svm_model.predict(feature)[0]
                    break

        return final_pred, base_pred, probs.cpu().numpy()