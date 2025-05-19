#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入模块
from data import DDoSDataset, create_dataloader
from model import BiLSTMDetector
from trainer import Trainer
import utils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ddos_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP =  {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'Portmap': 6, 'SNMP': 7, 'SSDP': 8, 'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}
CLASS_NAMES = list(CLASS_MAP.values())


def train_model(train_data_path, val_data_path, output_dir="./outputs",
                batch_size=256, epochs=10, learning_rate=0.001,
                weight_decay=0.001, gradient_clip=1.0):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置预处理器保存路径
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建数据集和数据加载器
    logger.info("加载数据集...")
    try:
        # 训练数据集 - 将拟合并保存预处理器
        train_dataset = DDoSDataset(
            data_path=train_data_path,
            preprocessor_path=preprocessor_path,  # 保存预处理器
            train=True
        )
        logger.info(f"训练数据集大小: {len(train_dataset)}")

        # 验证数据集 - 使用训练集拟合的预处理器
        val_dataset = DDoSDataset(
            data_path=val_data_path,
            preprocessor_path=preprocessor_path,  # 使用保存的预处理器
            train=False  # 验证模式
        )
        logger.info(f"验证数据集大小: {len(val_dataset)}")

        # 获取样本形状
        x_sample, y_sample = train_dataset[0]
        logger.info(f"样本形状: {x_sample.shape}, 标签形状: {y_sample.shape}")

        # 创建数据加载器
        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}")
        raise

    # 初始化模型
    logger.info("初始化模型...")
    input_size = 1  # 根据数据集: 样本特征形状为 [20, 1]
    num_classes = 13  # 根据标签映射


    model = BiLSTMDetector(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout_rate=0.5
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    # 训练模型
    logger.info("开始训练...")
    history = trainer.train(epochs=epochs, early_stopping_patience=10)

    # 绘制训练历史
    utils.plot_training_history(
        history=history,
        save_path=os.path.join(output_dir, "training_history.png")
    )

    # 加载最佳模型
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    trainer.load_checkpoint(best_model_path)

    # 在验证集上评估
    logger.info("在验证集上评估模型...")
    val_loss, val_acc, y_true, y_pred = utils.evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device
    )

    # 绘制混淆矩阵
    utils.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    # 获取分类报告
    report_df = utils.get_classification_report(y_true, y_pred, CLASS_NAMES)
    logger.info(f"分类报告:\n{report_df}")
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # 获取带概率的详细预测
    _, _, y_probs = utils.predict_batch(model, val_loader, device, return_probs=True)

    # 绘制ROC曲线
    # 首先将y_true转换为one-hot编码
    y_true_onehot = np.zeros((y_true.size, num_classes))
    y_true_onehot[np.arange(y_true.size), y_true] = 1

    utils.plot_roc_curves(
        y_true=y_true_onehot,
        y_score=y_probs,
        class_names=CLASS_NAMES,
        save_path=os.path.join(output_dir, "roc_curves.png")
    )

    # 可视化注意力权重
    logger.info("可视化注意力权重...")
    # 从验证集获取一个样本
    val_sample, val_label = val_dataset[0]
    # 添加批次维度
    val_sample = val_sample.unsqueeze(0)
    val_label = val_label.item()

    # 将模型导出为ONNX用于部署
    logger.info("导出模型为ONNX格式...")
    sample_input = torch.randn(1, 20, 1).to(device)  # 批次大小1，序列长度20，特征维度1
    onnx_path = os.path.join(output_dir, "ddos_detector.onnx")
    utils.export_model_onnx(model, sample_input, onnx_path)

    logger.info(f"模型训练和评估完成。结果保存到 {output_dir}")
    return model, history


def main():
    """运行训练和评估的主函数"""
    # 设置路径
    train_data_path = "C:\\Users\\17380\\train_dataset.csv"  # 替换为您的训练数据路径
    val_data_path = "C:\\Users\\17380\\test_dataset.csv"  # 替换为您的验证数据路径
    output_dir = "./outputs"

    # 训练和评估模型
    model, history = train_model(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        batch_size=128,
        epochs=8,
        learning_rate=0.001,
        weight_decay=0.001,
        gradient_clip=1.0
    )

    logger.info("DDoS检测系统训练成功完成！")


if __name__ == "__main__":
    main()