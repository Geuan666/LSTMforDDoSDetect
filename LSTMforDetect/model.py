# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.svm import SVC
import os
import pickle
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)

class BiLSTMDetector(nn.Module):  # 保持类名不变以确保兼容性
    """
    用于DDoS攻击检测的GRU模型（支持单向和双向）
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=13,
                 dropout_rate=0.2, bidirectional=False, model_type='gru'):  # 添加model_type参数
        """
        初始化DDoS检测模型
        参数:
            input_size: 每个时间步的输入特征数量
            hidden_size: 隐藏状态维度
            num_layers: 层数
            num_classes: 输出类别数量
            dropout_rate: Dropout概率
            bidirectional: 是否使用双向
            model_type: 模型类型 ('lstm' 或 'gru')
        """
        super(BiLSTMDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.model_type = model_type.lower()  # 新增属性

        # 根据model_type选择RNN层
        if self.model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:  # LSTM
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 计算RNN输出维度
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size

        # 层归一化层
        self.batch_norm = nn.BatchNorm1d(rnn_output_size)

        # 分类器
        self.fc1 = nn.Linear(rnn_output_size, hidden_size)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # 初始化权重
        self._init_weights()

        logger.info(f"初始化{model_type.upper()}Detector: input_size={input_size}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}, "
                    f"num_classes={num_classes}, dropout_rate={dropout_rate}, "
                    f"bidirectional={bidirectional}, model_type={model_type}")

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:  # 修改这里，统一处理RNN权重
                    # RNN权重: 使用正交初始化
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # 线性层权重: 使用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 1维权重: 使用常数初始化
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
        batch_size, seq_len, _ = x.shape

        # RNN前向传播
        if self.model_type == 'gru':
            # GRU只返回输出和最后的隐藏状态
            rnn_out, final_hidden = self.rnn(x)
        else:
            # LSTM返回输出、隐藏状态和细胞状态
            rnn_out, (final_hidden, _) = self.rnn(x)

        if self.bidirectional:
            # 双向的处理逻辑
            # final_hidden的形状: [num_layers*2, batch_size, hidden_size]
            final_hidden = final_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            final_forward = final_hidden[-1, 0, :, :]   # 前向最后一层
            final_backward = final_hidden[-1, 1, :, :]  # 后向最后一层
            combined = torch.cat((final_forward, final_backward), dim=1)
        else:
            # 单向的处理逻辑
            # final_hidden的形状: [num_layers, batch_size, hidden_size]
            combined = final_hidden[-1]  # 取最后一层

        # 应用层归一化
        combined = self.batch_norm(combined)

        # 应用dropout
        combined = self.dropout(combined)

        # 通过分类器
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)

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

class XGBoostModel:
    """XGBoost分类器模型类"""

    def __init__(self, class1, class2, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        初始化XGBoost模型

        参数:
            class1: 第一个类别的索引
            class2: 第二个类别的索引
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 子采样比例
            colsample_bytree: 特征子采样比例
            random_state: 随机种子
        """
        self.class1 = class1
        self.class2 = class2
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0  # 减少输出信息
        )

    def fit(self, X, y):
        """
        训练XGBoost模型

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

    def predict_proba(self, X):
        """
        预测样本概率

        参数:
            X: 特征矩阵

        返回:
            预测概率
        """
        return self.model.predict_proba(X)

    def save(self, path):
        """
        保存模型

        参数:
            path: 保存路径
        """
        model_data = {
            'model': self.model,
            'class1': self.class1,
            'class2': self.class2
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path):
        """
        加载模型

        参数:
            path: 模型路径

        返回:
            加载的XGBoost模型实例
        """
        model_data = joblib.load(path)
        instance = cls(model_data['class1'], model_data['class2'])
        instance.model = model_data['model']
        return instance

class SVMCascadeModel:
    """
    SVM级联模型：结合BiLSTM基础分类器和SVM二分类器
    """

    def __init__(self, base_model, confusion_pairs=None, confidence_threshold=0.95):
        """
        初始化级联模型
        参数:
            base_model: 基础BiLSTM模型
            confusion_pairs: 混淆类别对列表，如 [(11, 12), (5, 7), (2, 8), (9, 10), (0, 4), (0, 9)]
            confidence_threshold: 不触发二级分类器的置信度阈值
        """
        self.base_model = base_model
        self.confusion_pairs = confusion_pairs or [(11, 12), (5, 7), (2, 8), (9, 10), (0, 4), (0, 9)]
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

class XGBoostCascadeModel:
    """
    XGBoost级联模型：结合GRU/LSTM基础分类器和XGBoost二分类器
    """

    def __init__(self, base_model, confusion_pairs=None, confidence_threshold=0.95):
        """
        初始化级联模型
        参数:
            base_model: 基础GRU/LSTM模型
            confusion_pairs: 混淆类别对列表，如 [(11, 12), (5, 7), (2, 8), (9, 10), (0, 4), (0, 9)]
            confidence_threshold: 不触发二级分类器的置信度阈值
        """
        self.base_model = base_model
        self.confusion_pairs = confusion_pairs or [(11, 12), (5, 7), (2, 8), (9, 10), (0, 4), (0, 9)]
        self.confidence_threshold = confidence_threshold
        self.xgb_models = {}

    def load_xgb_models(self, model_dir):
        """
        加载XGBoost二分类器
        参数:
            model_dir: XGBoost模型目录
        """
        for class1, class2 in self.confusion_pairs:
            model_path = os.path.join(model_dir, f"xgb_model_{class1}_{class2}.pkl")
            if os.path.exists(model_path):
                self.xgb_models[(class1, class2)] = XGBoostModel.load(model_path)
                logger.info(f"已加载XGBoost分类器: {class1} vs {class2}")
            else:
                logger.warning(f"找不到XGBoost分类器: {class1} vs {class2}")

    def predict(self, inputs, device=None):
        """
        使用级联模型预测

        参数:
            inputs: 基础模型的输入数据，形状为[batch_size, feature_dim, 1]
            device: 计算设备

        返回:
            final_pred: 最终预测标签
            base_pred: 基础模型预测
            probs: 预测概率
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基础模型预测
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(inputs.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            base_pred = outputs.max(1)[1].cpu().numpy()
            confidence = probs.max(1)[0].cpu().numpy()

        # 如果没有加载XGBoost分类器，直接返回基础预测
        if not self.xgb_models:
            return base_pred, base_pred, probs.cpu().numpy()

        # 最终预测结果
        final_pred = base_pred.copy()

        # 提取特征（从输入的倒数第二维）
        features = inputs.squeeze(-1).cpu().numpy()

        # 对每个样本应用XGBoost二次分类
        for i, (pred, conf) in enumerate(zip(base_pred, confidence)):
            # 检查是否是混淆类别对
            for class1, class2 in self.confusion_pairs:
                if pred in [class1, class2] and conf < self.confidence_threshold:
                    # 找到对应的XGBoost分类器
                    xgb_model = self.xgb_models.get((class1, class2))
                    if xgb_model is None:
                        continue

                    # 获取当前样本的特征
                    feature = features[i:i + 1]

                    # 预测并更新结果
                    final_pred[i] = xgb_model.predict(feature)[0]
                    break

        return final_pred, base_pred, probs.cpu().numpy()