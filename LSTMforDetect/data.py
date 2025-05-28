# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
import warnings
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, Subset
import joblib
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'Portmap': 6, 'SNMP': 7, 'SSDP': 8,
             'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}
CLASS_NAMES = list(CLASS_MAP.keys())


class DataProcessor:
    """
    数据处理类，负责加载、清洗和特征工程
    """

    def __init__(self, data_path: str, n_workers: int = 4, n_components: int = 25):
        """
        初始化数据处理器
        Args:
            data_path: 数据文件路径
            n_workers: 并行处理的工作进程数
            n_components: PCA降维后的维度
        """
        self.data_path = data_path
        self.n_workers = n_workers

        # PCA相关参数
        self.n_components = n_components
        self.pca_model = None

        # 存储特征提取器
        self.scalers = {}
        self.encoders = {}

        # 使用CSV文件中的实际列名（精确匹配，包括前导空格）
        self.base_features = [
            ' Protocol',
            ' Flow Duration',
            ' Total Fwd Packets',
            ' Total Backward Packets',
            ' Fwd Packet Length Max',
            ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean',
            ' Fwd Packet Length Std',
            'Bwd Packet Length Max',
            ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean',
            ' Bwd Packet Length Std',
            ' Flow Packets/s',
            ' Flow IAT Max',
            'Fwd IAT Total',
            ' Fwd IAT Mean',
            ' Fwd IAT Std',
            ' Fwd IAT Max',
            ' Fwd IAT Min',
            'Bwd IAT Total',
            ' Bwd IAT Mean',
            ' Bwd IAT Std',
            ' Bwd IAT Max',
            ' Bwd IAT Min',
            'Fwd PSH Flags',
            ' Bwd PSH Flags',
            ' Fwd Header Length',
            ' Bwd Header Length',
            ' Min Packet Length',
            ' Packet Length Std',
            ' RST Flag Count',
            ' ACK Flag Count',
            ' URG Flag Count',
            ' CWE Flag Count',
            ' Average Packet Size',
            ' Avg Fwd Segment Size',
            ' Avg Bwd Segment Size',
            'Init_Win_bytes_forward',
            ' Init_Win_bytes_backward',
            ' act_data_pkt_fwd',
            'Active Mean',
            ' Active Max',
            ' Active Min',
            ' Inbound',
        ]

        # 需要进行对数转换的特征
        self.log_transform_features_base = [
            ' Flow Packets/s', ' Flow Duration'
        ]

        # 类别特征（注意前导空格）
        self.categorical_features_base = [' Protocol']

    def load_data(self) -> pd.DataFrame:
        """加载CSV数据文件，只读取需要的列"""
        try:
            logger.info(f"开始读取文件: {os.path.basename(self.data_path)}")

            # 先读取文件头，获取列名
            try:
                header_df = pd.read_csv(self.data_path, nrows=0)
            except Exception as e:
                logger.warning(f"默认引擎读取头部失败: {e}，切换 Python 引擎")
                header_df = pd.read_csv(self.data_path, nrows=0, engine='python')

            # 获取实际存在的列名
            available_columns = set(header_df.columns)
            logger.info(f"CSV文件包含 {len(available_columns)} 个列")

            # 确定要读取的列名
            usecols = []
            missing_features = []

            # 添加特征列
            for base_col in self.base_features:
                if base_col in available_columns:
                    usecols.append(base_col)
                else:
                    missing_features.append(base_col)

            # 添加标签列
            label_col = None
            if ' Label' in available_columns:
                label_col = ' Label'
                usecols.append(label_col)
            elif 'Label' in available_columns:
                label_col = 'Label'
                usecols.append(label_col)
            else:
                # 尝试其他方式找标签列
                for col in header_df.columns:
                    if 'label' in col.lower():
                        usecols.append(col)
                        label_col = col
                        logger.info(f"使用替代标签列: '{col}'")
                        break

            if not label_col:
                logger.error("无法找到标签列")
                return pd.DataFrame()

            if missing_features:
                logger.warning(f"以下 {len(missing_features)} 个特征在CSV中不存在：")
                for mf in missing_features[:10]:  # 只显示前10个
                    logger.warning(f"  缺失特征: '{mf}'")
                if len(missing_features) > 10:
                    logger.warning(f"  ... 还有 {len(missing_features) - 10} 个特征缺失")

            logger.info(f"将读取 {len(usecols)} 列: {len(usecols) - 1} 个特征列和 1 个标签列")

            # 分块读取，考虑到文件可能很大
            chunks = []
            try:
                chunk_iter = pd.read_csv(self.data_path, chunksize=10000, usecols=usecols, on_bad_lines='skip')
                for chunk in chunk_iter:
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"C 引擎读取失败: {e}，切换 Python 引擎")
                chunk_iter = pd.read_csv(self.data_path, engine='python', chunksize=10000, usecols=usecols,
                                         on_bad_lines='skip')
                for chunk in chunk_iter:
                    chunks.append(chunk)

            if not chunks:
                logger.error("读取数据失败，无数据块")
                return pd.DataFrame()

            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"文件 {os.path.basename(self.data_path)} 读取完成，shape={df.shape}")

            return df

        except Exception as e:
            logger.error(f"加载文件 {self.data_path} 时出错: {e}")
            return pd.DataFrame()

    def dropna_in_chunks(self, df, chunk_size=100000):
        """分块处理NaN值，避免内存不够"""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].dropna()
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗：去重、处理缺失值、异常值处理"""
        logger.info("开始数据清洗")
        logger.info(f"原始数据形状: {df.shape}")

        # 1. 移除重复记录
        df_clean = df.drop_duplicates()
        logger.info(f"移除重复记录后剩余 {len(df_clean)} 条记录")
        after_dedup = len(df_clean)

        # 获取标签列名
        label_col = None
        if ' Label' in df_clean.columns:
            label_col = ' Label'
        elif 'Label' in df_clean.columns:
            label_col = 'Label'
        else:
            # 尝试其他方式找标签列
            for col in df_clean.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: '{col}'")
                    break

        if not label_col:
            logger.error("清洗阶段无法找到标签列")
            return pd.DataFrame()

        # 2. 处理缺失值
        missing_before = df_clean.isnull().sum().sum()
        logger.info(f"处理前缺失值总数: {missing_before}")

        if missing_before > 0:
            df_clean = self.dropna_in_chunks(df_clean)
            logger.info(f"删除缺失值后剩余 {len(df_clean)} 条记录 (删除了 {after_dedup - len(df_clean)} 条)")

        # 对分类特征使用众数填充（如果还有缺失值）
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != label_col:  # 不处理标签列
                mode_values = df_clean[col].mode()
                if len(mode_values) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_values[0])

        # 3. 异常值处理（使用IQR方法）
        outlier_count = 0
        for col in numeric_cols:
            if col != label_col:  # 不处理标签列
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # 避免除零错误
                    lower_bound = Q1 - 5 * IQR
                    upper_bound = Q3 + 5 * IQR
                    # 统计异常值数量
                    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    outlier_count += outliers
                    # 将异常值限制在边界范围内
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info(f"处理了 {outlier_count} 个异常值")
        logger.info(f"数据清洗完成，最终形状: {df_clean.shape}")
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征预处理：独热编码、标准化、归一化、PCA降维"""

        logger.info("开始特征预处理")
        df_processed = df.copy()

        # 1: 处理类别特征（独热编码）
        for base_col in self.categorical_features_base:
            if base_col in df_processed.columns:
                already_encoded = any(col.startswith(f"{base_col.strip()}_") for col in df_processed.columns)
                if already_encoded:
                    logger.info(f"检测到特征 '{base_col}' 已经完成独热编码，跳过")
                    continue

                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[base_col]])
                    self.encoders[base_col] = encoder
                    logger.info(f"对特征 '{base_col}' 进行独热编码，生成 {encoded_data.shape[1]} 个新特征")
                else:
                    encoder = self.encoders.get(base_col)
                    if encoder is None:
                        logger.warning(f"找不到特征 '{base_col}' 的编码器，跳过处理")
                        continue
                    encoded_data = encoder.transform(df_processed[[base_col]])

                encoded_cols = [f"{base_col.strip()}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_processed.index)

                df_processed = df_processed.drop(base_col, axis=1)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)

        # 2: 获取标签列
        label_col = None
        if ' Label' in df_processed.columns:
            label_col = ' Label'
        elif 'Label' in df_processed.columns:
            label_col = 'Label'
        else:
            for col in df_processed.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: '{col}'")
                    break

        if not label_col:
            logger.error("预处理阶段无法找到标签列")
            raise ValueError("无法找到标签列")

        # 新增：对标签进行one-hot编码
        if fit:
            self.label_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            label_encoded = self.label_encoder.fit_transform(df_processed[[label_col]])
            self.label_classes = self.label_encoder.categories_[0]
            logger.info(f"标签类别: {self.label_classes}")
        else:
            if not hasattr(self, 'label_encoder'):
                raise ValueError("预测模式下缺少标签编码器")
            label_encoded = self.label_encoder.transform(df_processed[[label_col]])

        # 创建one-hot编码的列名
        label_columns = [f'label_{cls}' for cls in self.label_classes]
        label_df = pd.DataFrame(label_encoded, columns=label_columns, index=df_processed.index)

        # 保存原始标签（用于后续可能的分析）
        self.original_labels = df_processed[label_col].values

        # 从df_processed中删除原始标签列
        df_processed = df_processed.drop(columns=[label_col])

        # 3: 数值特征归一化
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)

        logger.info(f"找到 {len(numeric_cols)} 个数值特征")

        os.makedirs('models', exist_ok=True)  # 创建模型目录

        if fit:
            self.numeric_feature_order = numeric_cols
            self.minmax_scaler = MinMaxScaler()

            # 检查数值特征是否包含无效值
            for col in numeric_cols:
                if df_processed[col].isnull().any():
                    logger.warning(f"特征 '{col}' 包含NaN值，用0填充")
                    df_processed[col] = df_processed[col].fillna(0)
                if np.isinf(df_processed[col]).any():
                    logger.warning(f"特征 '{col}' 包含无穷值，用有限值替换")
                    df_processed[col] = np.where(np.isinf(df_processed[col]), 0, df_processed[col])

            df_processed[numeric_cols] = self.minmax_scaler.fit_transform(df_processed[numeric_cols])

            # 保存 scaler 和特征顺序
            joblib.dump(self.minmax_scaler, 'models/minmax_scaler.pkl')
            with open('models/numeric_feature_order.json', 'w') as f:
                json.dump(self.numeric_feature_order, f)
            logger.info(f"保存了 {len(numeric_cols)} 个数值特征的顺序和scaler")
        else:
            try:
                self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
                with open('models/numeric_feature_order.json', 'r') as f:
                    saved_feature_order = json.load(f)

                # 检查特征顺序是否匹配
                missing_features = []
                available_features = []

                for saved_col in saved_feature_order:
                    if saved_col in df_processed.columns:
                        available_features.append(saved_col)
                    else:
                        missing_features.append(saved_col)

                if missing_features:
                    logger.warning(f"缺失 {len(missing_features)} 个训练时使用的特征")

                self.numeric_feature_order = available_features
                numeric_cols = available_features

                if len(numeric_cols) == 0:
                    raise RuntimeError("无法找到任何匹配的数值特征列")

                logger.info(f"验证阶段使用 {len(numeric_cols)} 个特征")

                # 检查数值特征是否包含无效值
                for col in numeric_cols:
                    if df_processed[col].isnull().any():
                        logger.warning(f"特征 '{col}' 包含NaN值，用0填充")
                        df_processed[col] = df_processed[col].fillna(0)
                    if np.isinf(df_processed[col]).any():
                        logger.warning(f"特征 '{col}' 包含无穷值，用有限值替换")
                        df_processed[col] = np.where(np.isinf(df_processed[col]), 0, df_processed[col])

            except Exception as e:
                raise RuntimeError(f"验证阶段缺少 scaler 或特征顺序: {e}") from e

            df_processed[numeric_cols] = self.minmax_scaler.transform(df_processed[numeric_cols])

        # 保存归一化后、PCA前的特征数据（适用于SVM）
        self.normalized_features = df_processed[numeric_cols].values
        if label_col:
            self.normalized_labels = df_processed[label_col].values

        # 4: PCA 降维
        if fit:
            logger.info(f"执行 PCA 降维: 从 {len(numeric_cols)} 维降至 {self.n_components} 维")
            self.pca_model = PCA(n_components=self.n_components)
            pca_result = self.pca_model.fit_transform(df_processed[numeric_cols])
            explained_var = sum(self.pca_model.explained_variance_ratio_) * 100
            logger.info(f"PCA降维后保留信息量: {explained_var:.2f}%")

            # 保存 PCA 模型
            joblib.dump(self.pca_model, 'models/pca_model.pkl')
        else:
            try:
                self.pca_model = joblib.load('models/pca_model.pkl')
            except Exception as e:
                raise RuntimeError("验证阶段缺少 PCA 模型，并且加载失败") from e

            pca_result = self.pca_model.transform(df_processed[numeric_cols])

        # 5: 构造结果
        pca_columns = [f'pca_component_{i + 1}' for i in range(self.n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df_processed.index)

        # 保存PCA后的特征数据
        self.pca_features = pca_result

        if label_col:
            result_df = pd.concat([pca_df, df_processed[[label_col]]], axis=1)
        else:
            result_df = pca_df

        logger.info(f"PCA降维完成，最终特征维数: {result_df.shape[1] - (1 if label_col else 0)}")
        return result_df

    def process_data_pipeline(self, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """完整数据处理流水线"""
        logger.info(f"开始数据处理流水线，模式: {'训练' if train else '验证'}")

        # 1. 加载数据
        df = self.load_data()
        if df.empty:
            logger.error("数据加载失败")
            return np.array([]), np.array([])

        # 2. 数据清洗
        df_clean = self.clean_data(df)
        if df_clean.empty:
            logger.error("数据清洗后为空")
            return np.array([]), np.array([])

        # 特征预处理
        df_processed = self.preprocess_features(df_clean, fit=train)
        self.last_processed_df = df_processed.copy()

        # 提取特征（不包括标签列）
        feature_cols = [col for col in df_processed.columns if not col.startswith('label_')]
        label_cols = [col for col in df_processed.columns if col.startswith('label_')]

        X = df_processed[feature_cols].values
        y = df_processed[label_cols].values  # one-hot编码的标签

        # 检查特征是否有NaN或无穷值
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("特征数据中包含NaN或无穷值，将其替换为0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"最终特征形状: {X.shape}, 标签形状: {y.shape}")
        logger.info("数据处理流水线完成")

        return X, y, self.original_labels

    def save_preprocessors(self, save_path: str):
        """保存预处理器，包括PCA模型和标签编码器"""
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca_model': self.pca_model,
            'n_components': self.n_components,
            'minmax_scaler': self.minmax_scaler,
            'numeric_feature_order': self.numeric_feature_order,
            'base_features': self.base_features,
            'categorical_features_base': self.categorical_features_base,
            'label_encoder': self.label_encoder,
            'label_classes': self.label_classes
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        logger.info(f"预处理器已保存至 {save_path}")

    def load_preprocessors(self, load_path: str):
        """加载预处理器，包括PCA模型和标签编码器"""
        with open(load_path, 'rb') as f:
            preprocessors = pickle.load(f)

        self.scalers = preprocessors.get('scalers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.pca_model = preprocessors.get('pca_model')
        self.n_components = preprocessors.get('n_components', 20)
        self.minmax_scaler = preprocessors.get('minmax_scaler')
        self.numeric_feature_order = preprocessors.get('numeric_feature_order')
        self.label_encoder = preprocessors.get('label_encoder')  # 新增
        self.label_classes = preprocessors.get('label_classes')  # 新增

        # 兼容旧版本
        if 'base_features' in preprocessors:
            self.base_features = preprocessors['base_features']
        if 'categorical_features_base' in preprocessors:
            self.categorical_features_base = preprocessors['categorical_features_base']

        logger.info(f"预处理器已从 {load_path} 加载，PCA维度: {self.n_components}")

    def get_normalized_data(self):
        """获取归一化后的数据，用于SVM训练"""
        if hasattr(self, 'normalized_features') and hasattr(self, 'normalized_labels'):
            return self.normalized_features, self.normalized_labels
        else:
            logger.error("未找到归一化后的数据")
            return None, None

    def get_pca_data(self):
        """获取PCA降维后的数据，用于SVM训练"""
        if hasattr(self, 'pca_features') and hasattr(self, 'normalized_labels'):
            return self.pca_features, self.normalized_labels
        else:
            logger.error("未找到PCA降维后的数据")
            return None, None


class DDoSDataset(Dataset):
    """DDoS攻击预测的PyTorch数据集类"""

    def __init__(self, data_path: str, preprocessor_path: Optional[str] = None,
                 train: bool = True, transform: Optional[Any] = None):
        """初始化数据集"""
        self.transform = transform

        # 初始化处理器
        self.processor = DataProcessor(
            data_path=data_path,
            n_workers=1
        )

        # 处理数据
        if train and preprocessor_path:
            self.features, self.labels, self.label_names = self.processor.process_data_pipeline(train=True)
            logger.info(f"保存预处理器到: {preprocessor_path}")
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.processor.save_preprocessors(preprocessor_path)
            logger.info("预处理器保存成功")
        elif not train and preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"预处理器文件不存在: {preprocessor_path}")
            logger.info(f"加载预处理器从: {preprocessor_path}")
            self.processor.load_preprocessors(preprocessor_path)
            logger.info("预处理器加载成功")
            self.features, self.labels, self.label_names = self.processor.process_data_pipeline(train=False)
        else:
            self.features, self.labels, self.label_names = self.processor.process_data_pipeline(train=train)
            if train:
                logger.warning("训练模式未提供预处理器保存路径，将无法在预测时使用一致的预处理")

        # 确保数据不为空
        if len(self.features) == 0 or len(self.labels) == 0:
            raise ValueError("处理数据失败，未能生成有效的特征和标签")

        # 检查和打印特征和标签的数据类型
        logger.info(f"特征数据类型: {self.features.dtype}")
        logger.info(f"标签数据类型: {self.labels.dtype}")

        # 转换为PyTorch张量
        self.features = torch.from_numpy(self.features.astype(np.float32)).float()
        self.labels = torch.from_numpy(self.labels.astype(np.float32)).float()  # one-hot标签保持为float类型

        logger.info(f"特征形状: {self.features.shape}, 类型: {self.features.dtype}")
        logger.info(f"标签形状: {self.labels.shape}, 类型: {self.labels.dtype}")
        logger.info(f"标签类别数: {self.labels.shape[1] if len(self.labels.shape) > 1 else 1}")

        # 保存类别信息
        self.num_classes = self.labels.shape[1] if len(self.labels.shape) > 1 else 1
        self.label_classes = self.processor.label_classes if hasattr(self.processor, 'label_classes') else None


def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def main():
    # 设置日志级别以查看详细信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 删除旧的模型文件（如果存在）
    import shutil
    if os.path.exists('models'):
        shutil.rmtree('models')
        logger.info("删除了旧的模型文件")

    if os.path.exists('./outputs'):
        shutil.rmtree('./outputs')
        logger.info("删除了旧的输出文件")

    # 训练模式，生成新的预处理器
    logger.info("开始训练模式，生成新的预处理器...")
    try:
        train_dataset = DDoSDataset(
            data_path=r"C:\Users\17380\train_dataset.csv",
            preprocessor_path='./outputs/preprocessor.pkl',
            train=True,  # 训练模式
        )

        # 获取PCA数据
        X_pca, y = train_dataset.processor.get_pca_data()

        # 打印数据形状
        print("\n" + "=" * 50)
        print("数据处理结果:")
        print("=" * 50)
        print(f"X_pca shape: {X_pca.shape if X_pca is not None else 'None'}")
        print(f"y shape: {y.shape if y is not None else 'None'}")

        # 打印更多有用信息
        if X_pca is not None and y is not None:
            print(f"X_pca dtype: {X_pca.dtype}")
            print(f"y dtype: {y.dtype}")

            # 打印标签分布情况
            unique_labels, counts = np.unique(y, return_counts=True)
            print("\n标签分布情况:")
            for label, count in zip(unique_labels, counts):
                print(f"  类别 {label}: {count} 个样本 ({count / len(y) * 100:.2f}%)")

            # 打印前几个样本的标签
            print(f"\n前10个样本标签: {y[:10]}")

            # 打印一些统计信息
            print(f"\n特征数据统计:")
            print(f"  最小值: {X_pca.min():.4f}")
            print(f"  最大值: {X_pca.max():.4f}")
            print(f"  平均值: {X_pca.mean():.4f}")
            print(f"  标准差: {X_pca.std():.4f}")

            # 检查是否有异常值
            nan_count = np.isnan(X_pca).sum()
            inf_count = np.isinf(X_pca).sum()
            print(f"  NaN值数量: {nan_count}")
            print(f"  无穷值数量: {inf_count}")

            print("\n" + "=" * 50)
            print("数据处理成功完成！")
            print("=" * 50)
        else:
            print("数据处理失败！")

    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()