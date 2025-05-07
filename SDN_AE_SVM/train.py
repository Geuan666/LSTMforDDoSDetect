import os
import numpy as np
import torch
import time
from data import load_preprocessed_data, create_torch_datasets
from AE_SVM import AESVM

def train_model(data_folder, output_folder, params=None):
    """
    训练AE-SVM模型

    参数:
        data_folder: 预处理数据的文件夹路径
        output_folder: 模型和结果的输出文件夹路径
        params: 训练参数字典
    """
    start_time = time.time()

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 设置默认参数
    if params is None:
        params = {
            'hidden_dim': 25,  # 自编码器隐藏层维度
            'epochs': 110,  # 训练轮数
            'learning_rate': 0.0005,  # 学习率
            'batch_size': 64,  # 批大小
            'l2_weight': 0.0059,  # L2权重正则化参数
            'sparsity_param': 0.002,  # 稀疏性参数
            'sparsity_reg': 0.007,  # 稀疏性正则化参数
            'svm_kernel': 'rbf',  # SVM核函数
            'svm_C': 1.0,  # SVM正则化参数
            'svm_gamma': 'scale',  # SVM的gamma参数
        }

    print("加载预处理后的数据...")
    X_train, y_train, X_test, y_test, scaler = load_preprocessed_data(data_folder)

    print("创建数据加载器...")
    train_loader, test_loader, input_dim = create_torch_datasets(
        X_train, y_train, X_test, y_test, batch_size=params['batch_size']
    )

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化AE-SVM模型
    print("初始化AE-SVM模型...")
    model = AESVM(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        kernel=params['svm_kernel'],
        C=params['svm_C'],
        gamma=params['svm_gamma'],
        device=device
    )

    # 训练模型
    print("开始训练模型...")
    model.train(
        train_loader=train_loader,
        train_X=X_train,
        train_y=y_train,
        epochs=params['epochs'],
        learning_rate=params['learning_rate'],
        l2_weight=params['l2_weight'],
        sparsity_param=params['sparsity_param'],
        sparsity_reg=params['sparsity_reg']
    )

    # 保存模型
    model.save(os.path.join(output_folder, 'model'))

    # 计算训练时间
    training_time = time.time() - start_time
    print(f"模型训练完成! 耗时: {training_time:.2f}秒")

    return model, training_time


if __name__ == "__main__":
    # 训练模型
    train_model(
        data_folder="./preprocessed_data",
        output_folder="./results"
    )