import os
import numpy as np
import torch
import time
from data import load_preprocessed_data
from AE_SVM import AESVM
from utils import evaluate_model, print_metrics, plot_roc_curve


def test_model(model_folder, data_folder, output_folder, input_dim):
    """
    测试AE-SVM模型

    参数:
        model_folder: 模型文件夹路径
        data_folder: 预处理数据的文件夹路径
        output_folder: 结果输出文件夹路径
        input_dim: 输入特征维度
    """
    start_time = time.time()

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载测试数据
    print("加载预处理后的测试数据...")
    X_train, y_train, X_test, y_test, scaler = load_preprocessed_data(data_folder)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("加载AE-SVM模型...")
    model = AESVM(input_dim=input_dim, device=device)
    model.load(model_folder)

    # 测试模型
    print("开始测试模型...")
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # 评估性能
    metrics = evaluate_model(y_test, y_pred, y_score)
    print_metrics(metrics)

    # 绘制ROC曲线
    plot_roc_curve(y_test, y_score)

    # 计算测试时间
    testing_time = time.time() - start_time
    print(f"模型测试完成! 耗时: {testing_time:.2f}秒")

    # 将指标保存到文件
    with open(os.path.join(output_folder, 'metrics.txt'), 'w') as f:
        f.write("性能指标:\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}\n")
        if metrics['auc'] is not None:
            f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write("\n混淆矩阵:\n")
        f.write(f"真正例 (TP): {metrics['true_positive']}\n")
        f.write(f"真负例 (TN): {metrics['true_negative']}\n")
        f.write(f"假正例 (FP): {metrics['false_positive']}\n")
        f.write(f"假负例 (FN): {metrics['false_negative']}\n")
        f.write(f"\n测试时间: {testing_time:.2f}秒\n")

    return metrics, testing_time


if __name__ == "__main__":
    # 测试模型
    X_train = np.load("./preprocessed_data/X_train.npy")
    input_dim = X_train.shape[1]

    test_model(
        model_folder="./results/model",
        data_folder="./preprocessed_data",
        output_folder="./results",
        input_dim=input_dim
    )