#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import pickle
import logging
import sys
from sklearn.preprocessing import StandardScaler
from collections import Counter

# 导入自定义模块
sys.path.append(".")  # 确保可以导入当前目录下的模块
from model import BiLSTMDetector
from data import DataProcessor

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


class CascadePredictor:
    def __init__(self,
                 preprocessor_path="./outputs/preprocessor.pkl",
                 bilstm_model_path="./outputs/checkpoints/best_model.pth",
                 svm_classifiers_dir="./outputs/svm_classifiers",
                 confusion_pairs=[(11, 12), (5, 7), (2, 8), (9, 10)],
                 confidence_threshold=0.95):
        """
        初始化级联预测器

        参数:
            preprocessor_path: 预处理器路径
            bilstm_model_path: BiLSTM模型路径
            svm_classifiers_dir: SVM分类器目录
            confusion_pairs: 混淆类别对列表
            confidence_threshold: 置信度阈值
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confusion_pairs = confusion_pairs
        self.confidence_threshold = confidence_threshold

        # 加载预处理器
        logger.info(f"加载预处理器: {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        # 加载BiLSTM模型
        logger.info(f"加载BiLSTM模型: {bilstm_model_path}")
        checkpoint = torch.load(bilstm_model_path, map_location=self.device)

        # 初始化模型
        self.model = BiLSTMDetector(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            num_classes=13,
            dropout_rate=0.5
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载SVM分类器
        self.svm_classifiers = {}
        for class1, class2 in self.confusion_pairs:
            svm_path = os.path.join(svm_classifiers_dir, f"svm_classifier_{class1}_{class2}.pkl")
            if os.path.exists(svm_path):
                with open(svm_path, 'rb') as f:
                    self.svm_classifiers[(class1, class2)] = pickle.load(f)
                logger.info(f"已加载SVM分类器: {class1} vs {class2}")
            else:
                logger.warning(f"找不到SVM分类器: {class1} vs {class2}")

    def preprocess_data(self, csv_path):
        """
        预处理CSV数据

        参数:
            csv_path: CSV文件路径

        返回:
            bilstm_input: BiLSTM模型输入
            original_features: 原始特征
            flow_ids: 流量ID列表
        """
        logger.info(f"正在处理CSV文件: {csv_path}")

        # 创建数据处理器
        processor = DataProcessor(data_path=csv_path)

        # 加载数据
        df = processor.load_data()
        if df.empty:
            logger.error("加载数据失败")
            return None, None, None

        # 数据清洗
        df_clean = processor.clean_data(df)

        # 保存FlowID以便后续分析
        flow_id_col = None
        for possible_col in ['Flow ID', 'FlowID', 'Flow_ID']:
            if possible_col in df_clean.columns:
                flow_id_col = possible_col
                break

        flow_ids = []
        if flow_id_col:
            flow_ids = df_clean[flow_id_col].values

        # 提取特征 - 这是关键步骤，需要确保特征顺序与训练时一致
        # 获取数值列
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()

        # 移除标签列(如果存在)
        label_col = None
        for possible_label in ['Label', 'label']:
            if possible_label in numeric_cols:
                label_col = possible_label
                numeric_cols.remove(label_col)
                break

        # 应用相同的预处理步骤
        # 1. 保存原始数值特征
        original_features = df_clean[numeric_cols].values

        # 2. 应用标准化 - 使用训练时的scaler
        scaler = self.preprocessor.get('scalers', {}).get('minmax_scaler')
        if scaler:
            scaled_features = scaler.transform(original_features)
        else:
            # 如果没有找到scaler，使用默认标准化
            logger.warning("未找到训练时的scaler，使用默认标准化")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(original_features)

        # 3. 应用PCA变换 - 使用训练时的PCA模型
        pca_model = self.preprocessor.get('pca_model')
        if pca_model:
            pca_features = pca_model.transform(scaled_features)
        else:
            logger.warning("未找到训练时的PCA模型，跳过PCA转换")
            pca_features = scaled_features

        # 4. 准备BiLSTM输入
        # 转换为张量并调整形状为[batch_size, seq_len, input_size]
        pca_tensor = torch.from_numpy(pca_features).float()
        # 序列长度为PCA维度
        seq_len = pca_tensor.size(1)
        # 重塑为[batch_size, seq_len, 1]
        bilstm_input = pca_tensor.unsqueeze(-1)

        logger.info(f"数据预处理完成，样本数: {len(bilstm_input)}")
        return bilstm_input, original_features, flow_ids

    def predict(self, csv_path):
        """
        使用级联模型进行预测

        参数:
            csv_path: CSV文件路径

        返回:
            predictions: 预测结果字典
        """
        # 预处理数据
        bilstm_input, original_features, flow_ids = self.preprocess_data(csv_path)
        if bilstm_input is None:
            return None

        # 初始化结果
        results = {
            'flow_ids': flow_ids,
            'bilstm_predictions': [],
            'bilstm_confidence': [],
            'final_predictions': [],
            'svm_used': [],
            'prediction_labels': []
        }

        # BiLSTM预测
        logger.info("使用BiLSTM模型进行预测...")
        self.model.eval()
        batch_size = 128

        with torch.no_grad():
            for i in range(0, len(bilstm_input), batch_size):
                batch = bilstm_input[i:i + batch_size].to(self.device)
                outputs = self.model(batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # 获取预测和置信度
                preds = outputs.max(1)[1].cpu().numpy()
                confs = probs.max(1)[0].cpu().numpy()

                results['bilstm_predictions'].extend(preds)
                results['bilstm_confidence'].extend(confs)

        # 初始化最终预测
        results['final_predictions'] = results['bilstm_predictions'].copy()
        results['svm_used'] = [False] * len(results['bilstm_predictions'])

        # 对需要二次分类的样本使用SVM
        logger.info("对需要二次分类的样本使用SVM...")
        for i, (pred, conf) in enumerate(zip(results['bilstm_predictions'], results['bilstm_confidence'])):
            # 检查是否是混淆类别对且置信度低
            for class1, class2 in self.confusion_pairs:
                if pred in [class1, class2] and conf < self.confidence_threshold:
                    # 找到对应的SVM分类器
                    svm_info = self.svm_classifiers.get((class1, class2))
                    if svm_info is None:
                        continue

                    # 使用原始特征
                    orig_feature = original_features[i:i + 1]

                    # 应用特征选择
                    selector = svm_info['selector']
                    selected_feature = selector.transform(orig_feature)

                    # SVM预测
                    svm_model = svm_info['model']
                    binary_pred = svm_model.predict(selected_feature)[0]
                    classes = svm_info['classes']

                    # 更新预测
                    results['final_predictions'][i] = classes[1] if binary_pred == 1 else classes[0]
                    results['svm_used'][i] = True
                    break

        # 将数字标签转换为类别名称
        for pred in results['final_predictions']:
            results['prediction_labels'].append(REVERSE_CLASS_MAP.get(pred, f"Unknown-{pred}"))

        # 计算统计信息
        bilstm_counter = Counter(results['bilstm_predictions'])
        final_counter = Counter(results['final_predictions'])
        svm_used_count = sum(results['svm_used'])

        logger.info(f"BiLSTM预测分布: {dict(bilstm_counter)}")
        logger.info(f"最终预测分布: {dict(final_counter)}")
        logger.info(
            f"使用SVM次数: {svm_used_count}/{len(results['bilstm_predictions'])} ({svm_used_count / len(results['bilstm_predictions']) * 100:.2f}%)")

        # 对于混淆类别对，计算变化
        for class1, class2 in self.confusion_pairs:
            # 计数BiLSTM对这两个类的预测
            bilstm_class1 = sum(1 for p in results['bilstm_predictions'] if p == class1)
            bilstm_class2 = sum(1 for p in results['bilstm_predictions'] if p == class2)

            # 计数最终对这两个类的预测
            final_class1 = sum(1 for p in results['final_predictions'] if p == class1)
            final_class2 = sum(1 for p in results['final_predictions'] if p == class2)

            # 计算变化
            change_class1 = final_class1 - bilstm_class1
            change_class2 = final_class2 - bilstm_class2

            if bilstm_class1 > 0 or bilstm_class2 > 0:
                logger.info(f"类别 {REVERSE_CLASS_MAP[class1]} vs {REVERSE_CLASS_MAP[class2]}:")
                logger.info(f"  BiLSTM: {bilstm_class1}/{bilstm_class2}")
                logger.info(f"  最终: {final_class1}/{final_class2}")
                logger.info(f"  变化: {change_class1}/{change_class2}")

        return results


def save_results(results, output_path="prediction_results.csv"):
    """保存预测结果到CSV文件"""
    if results is None:
        logger.error("没有可保存的结果")
        return

    data = {
        'Flow_ID': results['flow_ids'],
        'BiLSTM_Prediction': [REVERSE_CLASS_MAP.get(p, f"Unknown-{p}") for p in results['bilstm_predictions']],
        'BiLSTM_Confidence': results['bilstm_confidence'],
        'SVM_Used': results['svm_used'],
        'Final_Prediction': results['prediction_labels']
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"预测结果已保存到: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DDoS流量级联预测')
    parser.add_argument('--csv', default="C:\\Users\\17380\\test_dataset.csv", help='需要预测的CSV文件路径')
    parser.add_argument('--output', default='prediction_results.csv', help='输出结果CSV文件路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--bilstm', default='./outputs/checkpoints/best_model.pth', help='BiLSTM模型路径')
    parser.add_argument('--svm_dir', default='./outputs/svm_classifiers', help='SVM分类器目录')
    parser.add_argument('--threshold', type=float, default=0.95, help='置信度阈值')

    args = parser.parse_args()

    # 创建预测器
    predictor = CascadePredictor(
        preprocessor_path=args.preprocessor,
        bilstm_model_path=args.bilstm,
        svm_classifiers_dir=args.svm_dir,
        confidence_threshold=args.threshold
    )

    # 进行预测
    results = predictor.predict(args.csv)

    # 保存结果
    save_results(results, args.output)


if __name__ == "__main__":
    main()