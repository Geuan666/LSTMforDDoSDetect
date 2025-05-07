from sklearn import svm
import joblib
import numpy as np
import torch

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """初始化SVM分类器"""
        self.classifier = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True, verbose=True)

    def train(self, features, labels):
        """训练SVM分类器"""
        print(f"开始训练SVM分类器，特征维度: {features.shape}")
        self.classifier.fit(features, labels)
        print("SVM分类器训练完成")
        return self

    def predict(self, features):
        """使用SVM进行预测"""
        return self.classifier.predict(features)

    def predict_proba(self, features):
        """预测概率"""
        return self.classifier.predict_proba(features)

    def save(self, path):
        """保存模型"""
        joblib.dump(self.classifier, path)
        print(f"SVM模型已保存到: {path}")

    def load(self, path):
        """加载模型"""
        self.classifier = joblib.load(path)
        print(f"SVM模型已从 {path} 加载")
        return self