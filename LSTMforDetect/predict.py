#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import pickle
import logging
import sys
from collections import Counter

# 导入自定义模块
sys.path.append(".")  # 确保可以导入当前目录下的模块
from model import BiLSTMDetector, SVMCascadeModel
from data import DDoSDataset, create_dataloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5,
             'Portmap': 6, 'SNMP': 7, 'SSDP': 8, 'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def predict_with_cascade_model(csv_path, output_path=None,
                               preprocessor_path="./outputs/preprocessor.pkl",
                               lstm_model_path="./outputs/checkpoints/best_model.pth",
                               svm_models_dir="./outputs/svm_models",
                               confusion_pairs=None):
    """
    使用级联模型进行预测

    参数:
        csv_path: 要预测的CSV文件路径
        output_path: 输出结果的CSV文件路径
        preprocessor_path: 预处理器路径
        lstm_model_path: LSTM模型路径
        svm_models_dir: SVM模型目录
        confusion_pairs: 混淆类别对列表

    返回:
        results: 预测结果
    """
    if confusion_pairs is None:
        confusion_pairs = [(11, 12), (5, 7), (2, 8), (9, 10)]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info(f"开始处理CSV文件: {csv_path}")
    try:
        # 创建数据集
        test_dataset = DDoSDataset(
            data_path=csv_path,
            preprocessor_path=preprocessor_path,
            train=False
        )
        logger.info(f"数据集大小: {len(test_dataset)}")

        # 创建数据加载器
        test_loader = create_dataloader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        # 尝试获取Flow ID
        try:
            # 从原始数据中获取Flow ID
            flow_ids = test_dataset.processor.load_data().get('Flow ID', None)
            if flow_ids is None:
                logger.warning("无法获取Flow ID")
                flow_ids = [f"flow_{i}" for i in range(len(test_dataset))]
        except Exception as e:
            logger.warning(f"获取Flow ID时出错: {e}")
            flow_ids = [f"flow_{i}" for i in range(len(test_dataset))]

    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return None

    # 加载LSTM模型
    logger.info(f"加载LSTM模型: {lstm_model_path}")
    try:
        checkpoint = torch.load(lstm_model_path, map_location=device)

        # 初始化模型
        lstm_model = BiLSTMDetector(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            num_classes=13,
            dropout_rate=0.5
        )
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.to(device)
        lstm_model.eval()
    except Exception as e:
        logger.error(f"加载LSTM模型时出错: {e}")
        return None

    # 创建级联模型
    logger.info("创建级联模型...")
    cascade_model = SVMCascadeModel(
        base_model=lstm_model,
        confusion_pairs=confusion_pairs,
        confidence_threshold=0.95
    )

    # 加载SVM模型
    cascade_model.load_svm_models(svm_models_dir)

    # 进行预测
    logger.info("开始预测...")
    predictions = {
        'flow_id': flow_ids,
        'lstm_prediction': [],
        'lstm_confidence': [],
        'final_prediction': [],
        'svm_used': [],
        'prediction_label': []
    }

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            # 预测
            final_pred, base_pred, probs = cascade_model.predict(inputs, device)

            # 提取置信度
            confidence = probs.max(1)[0].cpu().numpy()

            # 记录是否使用了SVM
            batch_size = inputs.size(0)
            svm_used = np.zeros(batch_size, dtype=bool)

            # 检查哪些样本使用了SVM
            for j, (pred, conf) in enumerate(zip(base_pred, confidence)):
                for class1, class2 in confusion_pairs:
                    if pred in [class1, class2] and conf < 0.95:
                        if final_pred[j] != base_pred[j]:
                            svm_used[j] = True
                        break

            # 累积结果
            predictions['lstm_prediction'].extend(base_pred)
            predictions['lstm_confidence'].extend(confidence)
            predictions['final_prediction'].extend(final_pred)
            predictions['svm_used'].extend(svm_used)
            predictions['prediction_label'].extend([REVERSE_CLASS_MAP.get(p, f"Unknown-{p}") for p in final_pred])

            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                logger.info(f"批次: {i + 1}/{len(test_loader)}")

    # 确保所有列长度一致
    min_len = min(len(arr) for arr in predictions.values())
    for key in predictions:
        predictions[key] = predictions[key][:min_len]

    # 打印统计信息
    lstm_counter = Counter(predictions['lstm_prediction'])
    final_counter = Counter(predictions['final_prediction'])
    svm_used_count = sum(predictions['svm_used'])

    logger.info(f"LSTM预测分布: {dict(lstm_counter)}")
    logger.info(f"最终预测分布: {dict(final_counter)}")
    logger.info(
        f"使用SVM次数: {svm_used_count}/{len(predictions['lstm_prediction'])} ({svm_used_count / len(predictions['lstm_prediction']) * 100:.2f}%)")

    # 对于混淆类别对，计算变化
    for class1, class2 in confusion_pairs:
        # 计数LSTM对这两个类的预测
        lstm_class1 = sum(1 for p in predictions['lstm_prediction'] if p == class1)
        lstm_class2 = sum(1 for p in predictions['lstm_prediction'] if p == class2)

        # 计数最终对这两个类的预测
        final_class1 = sum(1 for p in predictions['final_prediction'] if p == class1)
        final_class2 = sum(1 for p in predictions['final_prediction'] if p == class2)

        # 计算变化
        change_class1 = final_class1 - lstm_class1
        change_class2 = final_class2 - lstm_class2

        if lstm_class1 > 0 or lstm_class2 > 0:
            logger.info(f"类别 {REVERSE_CLASS_MAP[class1]} vs {REVERSE_CLASS_MAP[class2]}:")
            logger.info(f"  LSTM: {lstm_class1}/{lstm_class2}")
            logger.info(f"  最终: {final_class1}/{final_class2}")
            logger.info(f"  变化: {change_class1}/{change_class2}")

    # 保存结果
    if output_path:
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        logger.info(f"预测结果已保存到: {output_path}")

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DDoS流量级联预测')
    parser.add_argument('--csv', default="C:\\Users\\17380\\test_dataset.csv", help='需要预测的CSV文件路径')
    parser.add_argument('--output', default='prediction_results.csv', help='输出结果CSV文件路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--lstm', default='./outputs/checkpoints/best_model.pth', help='LSTM模型路径')
    parser.add_argument('--svm_dir', default='./outputs/svm_models', help='SVM模型目录')

    args = parser.parse_args()

    # 执行预测
    predict_with_cascade_model(
        csv_path=args.csv,
        output_path=args.output,
        preprocessor_path=args.preprocessor,
        lstm_model_path=args.lstm,
        svm_models_dir=args.svm_dir
    )


if __name__ == "__main__":
    main()