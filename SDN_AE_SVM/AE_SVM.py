import torch
import numpy as np
import os
from AE import Autoencoder
from SVM import SVMClassifier


class AESVM:
    def __init__(self, input_dim, hidden_dim=25, kernel='rbf', C=1.0, gamma='scale', device='cuda'):
        """
        初始化AE-SVM模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: 自编码器隐藏层维度
            kernel: SVM核函数
            C: SVM正则化参数
            gamma: SVM的gamma参数
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化自编码器
        self.autoencoder = Autoencoder(input_dim, hidden_dim)

        # 初始化SVM分类器
        self.svm = SVMClassifier(kernel=kernel, C=C, gamma=gamma)

    def train(self, train_loader, train_X, train_y,
              epochs=100, learning_rate=0.001,
              l2_weight=0.0059, sparsity_param=0.002, sparsity_reg=0.007):
        """
        训练AE-SVM模型

        参数:
            train_loader: PyTorch数据加载器，用于训练自编码器
            train_X: 训练特征，用于SVM训练
            train_y: 训练标签，用于SVM训练
            epochs: 自编码器训练的轮数
            learning_rate: 学习率
            l2_weight: L2权重正则化参数
            sparsity_param: 稀疏性参数
            sparsity_reg: 稀疏性正则化参数
        """
        print("第1步: 训练自编码器...")
        self.autoencoder.train_model(
            train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            sparsity_param=sparsity_param,
            sparsity_reg=sparsity_reg,
            device=self.device
        )

        print("第2步: 使用自编码器提取特征...")
        # 将数据转换为PyTorch张量并移至设备
        train_tensor = torch.FloatTensor(train_X).to(self.device)

        # 提取编码特征
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encode(train_tensor).cpu().numpy()

        print("第3步: 使用编码特征训练SVM...")
        self.svm.train(encoded_features, train_y)

        print("AE-SVM模型训练完成!")
        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测

        参数:
            X: 输入特征

        返回:
            预测标签
        """
        # 将数据转换为PyTorch张量并移至设备
        test_tensor = torch.FloatTensor(X).to(self.device)

        # 提取编码特征
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encode(test_tensor).cpu().numpy()

        # 使用SVM进行预测
        return self.svm.predict(encoded_features)

    def predict_proba(self, X):
        """预测概率"""
        # 将数据转换为PyTorch张量并移至设备
        test_tensor = torch.FloatTensor(X).to(self.device)

        # 提取编码特征
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encode(test_tensor).cpu().numpy()

        # 使用SVM进行概率预测
        return self.svm.predict_proba(encoded_features)

    def save(self, folder):
        """
        保存模型到指定文件夹

        参数:
            folder: 保存模型的文件夹路径
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # 保存自编码器
        torch.save(self.autoencoder.state_dict(), os.path.join(folder, 'autoencoder.pth'))

        # 保存SVM
        self.svm.save(os.path.join(folder, 'svm_classifier.pkl'))

        print(f"模型已保存到: {folder}")

    def load(self, folder):
        """
        从指定文件夹加载模型

        参数:
            folder: 保存模型的文件夹路径
        """
        # 加载自编码器
        self.autoencoder.load_state_dict(torch.load(os.path.join(folder, 'autoencoder.pth')))
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        # 加载SVM
        self.svm.load(os.path.join(folder, 'svm_classifier.pkl'))

        print(f"模型已从 {folder} 加载")
        return self