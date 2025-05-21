#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import logging
import pickle

logger = logging.getLogger(__name__)


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
        self.svm_classifiers = {}

    def load_svm_classifiers(self, model_dir):
        """
        加载SVM二分类器
        参数:
            model_dir: SVM模型目录
        """
        for class1, class2 in self.confusion_pairs:
            model_path = os.path.join(model_dir, f"svm_classifier_{class1}_{class2}.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.svm_classifiers[(class1, class2)] = pickle.load(f)
                logger.info(f"已加载SVM分类器: {class1} vs {class2}")
            else:
                logger.warning(f"找不到SVM分类器: {class1} vs {class2}")

    def predict(self, inputs, pre_pca_features=None, device=None):
        """
        使用级联模型预测

        参数:
            inputs: BiLSTM模型的输入数据
            pre_pca_features: 用于SVM的原始特征
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

        # 如果没有提供原始特征或没有加载SVM分类器，直接返回基础预测
        if pre_pca_features is None or not self.svm_classifiers:
            return base_pred, base_pred, probs.cpu().numpy()

        # 最终预测结果
        final_pred = base_pred.copy()

        # 对每个样本应用SVM二次分类
        for i, (pred, conf) in enumerate(zip(base_pred, confidence)):
            # 检查是否是混淆类别对
            for class1, class2 in self.confusion_pairs:
                if pred in [class1, class2] and conf < self.confidence_threshold:
                    # 找到对应的SVM分类器
                    svm_info = self.svm_classifiers.get((class1, class2))
                    if svm_info is None:
                        continue

                    # 获取当前样本的原始特征
                    orig_feature = pre_pca_features[i:i + 1]

                    # 应用特征选择
                    selector = svm_info['selector']
                    selected_feature = selector.transform(orig_feature)

                    # SVM预测
                    svm_model = svm_info['model']
                    binary_pred = svm_model.predict(selected_feature)[0]
                    classes = svm_info['classes']

                    # 更新预测
                    final_pred[i] = classes[1] if binary_pred == 1 else classes[0]
                    break

        return final_pred, base_pred, probs.cpu().numpy()