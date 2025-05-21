#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from utils import CLASS_MAP, CLASS_NAMES  # 如果CLASS_MAP定义在utils.py中

logger = logging.getLogger(__name__)


def train_svm_classifier(data_path, class1, class2, output_path, test_size=0.2):
    """
    训练SVM二分类器

    参数:
        data_path: PCA前数据的路径
        class1, class2: 需要区分的两个类别
        output_path: 保存分类器的路径
        test_size: 测试集比例
    """
    logger.info(f"开始训练SVM二分类器: {class1} vs {class2}")

    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    # 获取类别名称（用于字符串标签）
    class1_name = None
    class2_name = None
    for name, idx in CLASS_MAP.items():
        if idx == class1:
            class1_name = name
        if idx == class2:
            class2_name = name

    # 创建标签掩码，同时考虑数字和字符串标签
    mask = np.zeros(len(labels), dtype=bool)
    for i, label in enumerate(labels):
        # 检查数字标签
        if isinstance(label, (int, np.integer)) and (label == class1 or label == class2):
            mask[i] = True
        # 检查字符串标签
        elif isinstance(label, str) and (label == class1_name or label == class2_name):
            mask[i] = True

    X = features[mask]
    y_raw = labels[mask]

    # 将标签统一转换为二分类(0,1)
    y_binary = np.zeros(len(y_raw), dtype=int)
    for i, label in enumerate(y_raw):
        if isinstance(label, (int, np.integer)) and label == class2:
            y_binary[i] = 1
        elif isinstance(label, str) and label == class2_name:
            y_binary[i] = 1

    logger.info(f"类别 {class1} 样本数: {sum(y_binary == 0)}")
    logger.info(f"类别 {class2} 样本数: {sum(y_binary == 1)}")

    # 检查样本数量
    if len(X) == 0 or sum(y_binary == 0) == 0 or sum(y_binary == 1) == 0:
        logger.error(f"没有足够的样本进行训练，跳过 {class1} vs {class2} 分类器")
        return None

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
    )

    # 特征选择
    k = min(X.shape[1] // 2, 50)  # 选择一半的特征或最多50个
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    logger.info(f"特征选择后的特征数: {X_train_selected.shape[1]}")

    # 定义SVM模型，严格使用RBF核函数
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')

    # 参数网格搜索
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(
        svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    logger.info("开始网格搜索SVM最优参数...")
    grid_search.fit(X_train_selected, y_train)

    best_svm = grid_search.best_estimator_
    logger.info(f"最优参数: {grid_search.best_params_}")
    logger.info(f"交叉验证最优分数: {grid_search.best_score_:.4f}")

    # 在测试集上评估
    y_pred = best_svm.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"测试集准确率: {accuracy:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # 保存模型和选择器
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_data = {
        'model': best_svm,
        'selector': selector,
        'classes': [class1, class2],
        'accuracy': accuracy,
        'best_params': grid_search.best_params_
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"SVM模型已保存到: {output_path}")

    return model_data


def load_svm_classifier(model_path):
    """加载SVM二分类器"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data