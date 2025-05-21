# -*- coding: utf-8 -*-
import os
import logging
import torch
import numpy as np
# 导入模块
from data import DDoSDataset, create_dataloader
from model import BiLSTMDetector, SVMModel, SVMCascadeModel
from trainer import LSTMTrainer, SVMTrainer
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
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'Portmap': 6, 'SNMP': 7, 'SSDP': 8,
             'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}
CLASS_NAMES = list(CLASS_MAP.keys())


def train_lstm_model(train_data_path, val_data_path, output_dir="./outputs",
                     batch_size=256, epochs=10, learning_rate=0.001,
                     weight_decay=0.001, gradient_clip=1.0):
    """
    训练LSTM模型

    参数:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        output_dir: 输出目录
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        gradient_clip: 梯度裁剪值

    返回:
        model: 训练好的模型
        history: 训练历史
    """
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
    logger.info("初始化LSTM模型...")
    input_size = 1  # 根据数据集: 样本特征形状为 [25, 1]
    num_classes = 13  # 根据标签映射

    model = BiLSTMDetector(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout_rate=0.5
    )

    # 初始化训练器
    trainer = LSTMTrainer(
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
    logger.info("开始训练LSTM模型...")
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
    logger.info("在验证集上评估LSTM模型...")
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

    # 将模型导出为ONNX用于部署
    logger.info("导出模型为ONNX格式...")
    sample_input = torch.randn(1, 25, 1).to(device)  # 批次大小1，序列长度25，特征维度1
    onnx_path = os.path.join(output_dir, "ddos_detector.onnx")
    utils.export_model_onnx(model, sample_input, onnx_path)

    logger.info(f"LSTM模型训练和评估完成。结果保存到 {output_dir}")
    return model, history


def train_svm_models(train_data_path, output_dir="./outputs", confusion_pairs=None):
    """
    训练SVM模型

    参数:
        train_data_path: 训练数据路径
        output_dir: 输出目录
        confusion_pairs: 混淆类别对列表

    返回:
        results: 训练结果
    """
    if confusion_pairs is None:
        confusion_pairs = [(11, 12), (5, 7), (2, 8), (9, 10)]

    # 创建SVM输出目录
    svm_output_dir = os.path.join(output_dir, "svm_models")
    os.makedirs(svm_output_dir, exist_ok=True)

    # 加载训练数据集
    logger.info("加载数据集用于SVM训练...")
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")

    try:
        train_dataset = DDoSDataset(
            data_path=train_data_path,
            preprocessor_path=preprocessor_path,
            train=False  # 使用已有的预处理器
        )
        logger.info(f"SVM训练数据集大小: {len(train_dataset)}")

        # 从数据处理器获取PCA降维后的数据
        X_pca, y = train_dataset.processor.get_pca_data()
        if X_pca is None or y is None:
            logger.error("无法获取PCA数据，使用处理后的特征进行训练")
            # 从数据集中构建训练数据
            X_pca = train_dataset.features.numpy()
            y = train_dataset.labels.squeeze().numpy()

        # 分析和打印标签分布情况
        if isinstance(y[0], str):
            logger.info("检测到字符串标签，统计各类别数量...")
            unique_labels, counts = np.unique(y, return_counts=True)
            labels_count = {label: count for label, count in zip(unique_labels, counts)}
            logger.info(f"标签分布: {labels_count}")

            # 打印混淆类别对对应的字符串标签名称
            for class1, class2 in confusion_pairs:
                class1_name = CLASS_NAMES[class1] if class1 < len(CLASS_NAMES) else f"Class-{class1}"
                class2_name = CLASS_NAMES[class2] if class2 < len(CLASS_NAMES) else f"Class-{class2}"
                logger.info(f"混淆类别对 {class1} vs {class2} 对应字符串标签: {class1_name} vs {class2_name}")
        else:
            logger.info("检测到数字标签，统计各类别数量...")
            unique_labels, counts = np.unique(y, return_counts=True)
            logger.info(f"标签分布: {dict(zip(unique_labels, counts))}")

        # 初始化SVM训练器
        svm_trainer = SVMTrainer(output_dir=svm_output_dir)

        # 训练多个SVM分类器
        logger.info("开始训练SVM分类器...")
        results = svm_trainer.train_multiple_classifiers(X_pca, y, confusion_pairs)

        # 打印训练结果
        for (class1, class2), accuracy in results.items():
            class1_name = CLASS_NAMES[class1] if class1 < len(CLASS_NAMES) else f"Class-{class1}"
            class2_name = CLASS_NAMES[class2] if class2 < len(CLASS_NAMES) else f"Class-{class2}"
            logger.info(f"SVM分类器 {class1_name} vs {class2_name} 准确率: {accuracy:.4f}")

        return results

    except Exception as e:
        logger.error(f"SVM训练出错: {str(e)}")
        raise


def evaluate_cascade_model(lstm_model, val_data_path, output_dir="./outputs", confusion_pairs=None):
    """
    评估级联模型

    参数:
        lstm_model: 训练好的LSTM模型
        val_data_path: 验证数据路径
        output_dir: 输出目录
        confusion_pairs: 混淆类别对列表

    返回:
        cascade_model: 级联模型
    """
    if confusion_pairs is None:
        confusion_pairs = [(11, 12), (5, 7), (2, 8), (9, 10)]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载验证数据集
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    val_dataset = DDoSDataset(
        data_path=val_data_path,
        preprocessor_path=preprocessor_path,
        train=False
    )
    val_loader = create_dataloader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 创建级联模型
    logger.info("初始化级联模型...")
    cascade_model = SVMCascadeModel(
        base_model=lstm_model,
        confusion_pairs=confusion_pairs,
        confidence_threshold=0.95
    )

    # 加载SVM模型
    svm_models_dir = os.path.join(output_dir, "svm_models")
    cascade_model.load_svm_models(svm_models_dir)

    # 评估级联模型
    logger.info("评估级联模型...")
    final_pred, base_pred, probs = cascade_model.predict(
        next(iter(val_loader))[0],  # 取一个批次的数据
        device=device
    )

    # 进行完整评估
    logger.info("在验证集上进行完整评估...")
    all_final_pred = []
    all_base_pred = []
    all_true = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            final_pred, base_pred, _ = cascade_model.predict(inputs, device)
            all_final_pred.extend(final_pred)
            all_base_pred.extend(base_pred)
            all_true.extend(targets.squeeze().cpu().numpy())

    # 计算准确率
    base_acc = np.mean(np.array(all_base_pred) == np.array(all_true))
    cascade_acc = np.mean(np.array(all_final_pred) == np.array(all_true))

    logger.info(f"LSTM基础模型准确率: {base_acc:.4f}")
    logger.info(f"级联模型准确率: {cascade_acc:.4f}")
    logger.info(f"精度提升: {(cascade_acc - base_acc) * 100:.2f}%")

    # 绘制混淆矩阵
    utils.plot_confusion_matrix(
        y_true=all_true,
        y_pred=all_final_pred,
        class_names=CLASS_NAMES,
        save_path=os.path.join(output_dir, "cascade_confusion_matrix.png")
    )

    # 针对混淆类别对分析
    for class1, class2 in confusion_pairs:
        # 筛选属于这两个类别的样本
        mask = np.logical_or(np.array(all_true) == class1, np.array(all_true) == class2)
        if sum(mask) == 0:
            continue

        subset_true = np.array(all_true)[mask]
        subset_base = np.array(all_base_pred)[mask]
        subset_cascade = np.array(all_final_pred)[mask]

        base_subset_acc = np.mean(subset_base == subset_true)
        cascade_subset_acc = np.mean(subset_cascade == subset_true)

        logger.info(f"类别 {CLASS_NAMES[class1]} vs {CLASS_NAMES[class2]}:")
        logger.info(f"  LSTM准确率: {base_subset_acc:.4f}")
        logger.info(f"  级联准确率: {cascade_subset_acc:.4f}")
        logger.info(f"  精度提升: {(cascade_subset_acc - base_subset_acc) * 100:.2f}%")

    return cascade_model


def main():
    """运行训练和评估的主函数"""
    # 设置路径
    train_data_path = "C:\\Users\\17380\\train_dataset.csv"  # 替换为您的训练数据路径
    val_data_path = "C:\\Users\\17380\\test_dataset.csv"  # 替换为您的验证数据路径
    output_dir = "./outputs"

    # 1. 训练LSTM模型
    lstm_model, history = train_lstm_model(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        batch_size=128,
        epochs=8,
        learning_rate=0.001,
        weight_decay=0.001,
        gradient_clip=1.0
    )

    # 2. 训练SVM模型（针对容易混淆的类别对）
    confusion_pairs = [(11, 12), (5, 7), (2, 8), (9, 10)]
    svm_results = train_svm_models(
        train_data_path=train_data_path,
        output_dir=output_dir,
        confusion_pairs=confusion_pairs
    )

    # 3. 评估级联模型
    cascade_model = evaluate_cascade_model(
        lstm_model=lstm_model,
        val_data_path=val_data_path,
        output_dir=output_dir,
        confusion_pairs=confusion_pairs
    )

    logger.info("DDoS检测系统训练成功完成！")


if __name__ == "__main__":
    main()