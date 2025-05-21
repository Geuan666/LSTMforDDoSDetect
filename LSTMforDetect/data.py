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
        self.column_map = None  # 用于存储列名映射

        # PCA相关参数
        self.n_components = n_components
        self.pca_model = None

        # 存储特征提取器
        self.scalers = {}
        self.encoders = {}

        # 不带空格版本的特征列表，使用TimeStamp的目的是防止删除重复值时删除过多，更好的解决方案是重新提取数据集
        self.base_features = [
            'Protocol', 'FlowDuration', 'TotalFwdPackets', 'TotalBackwardPackets',
            'FlowBytes/s', 'FlowPackets/s', 'FwdPacketLengthMax', 'FwdPacketLengthMin',
            'FwdPacketLengthMean', 'FwdPacketLengthStd', 'BwdPacketLengthMax',
            'BwdPacketLengthMin', 'BwdPacketLengthMean', 'BwdPacketLengthStd',
            'PacketLengthVariance', 'FlowIATMin', 'FlowIATMax', 'FlowIATMean',
            'FlowIATStd', 'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin',
            'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags',
            'BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags', 'FwdHeaderLength',
            'BwdHeaderLength', 'FwdPackets/s', 'BwdPackets/s', 'Init_Win_bytes_forward',
            'Init_Win_bytes_backward', 'min_seg_size_forward', 'SubflowFwdBytes',
            'SubflowBwdBytes', 'AveragePacketSize', 'AvgFwdSegmentSize',
            'AvgBwdSegmentSize', 'ActiveMean', 'ActiveMin', 'ActiveMax', 'ActiveStd',
            'IdleMean', 'IdleMin', 'IdleMax', 'IdleStd', 'Timestamp',
        ]

        # 需要进行对数转换的特征,经实验，使用对数转换效果较好
        self.log_transform_features_base = [
            'FlowBytes/s', 'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s',
            'FlowDuration', 'PacketLengthVariance'
        ]

        # 类别特征
        self.categorical_features_base = ['Protocol']

    def normalize_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        创建标准化的列名映射，将所有列名无空格版本作为键，原始列名作为值，原始数据 列名中存在不确定的前导空格
        """
        column_map = {}
        for col in df.columns:
            # 移除所有空格后的列名作为键
            normalized_key = col.replace(" ", "")
            column_map[normalized_key] = col

        logger.info(f"创建了列名映射，共 {len(column_map)} 个列")
        return column_map

    def get_actual_column_name(self, normalized_name: str) -> Optional[str]:
        """根据标准化名称获取数据集中的实际列名"""
        if self.column_map is None:
            logger.warning("列名映射尚未初始化")
            return None
        return self.column_map.get(normalized_name)

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

            # 先创建临时列名映射，用于识别需要的列
            temp_column_map = {}
            for col in header_df.columns:
                normalized_key = col.replace(" ", "")
                temp_column_map[normalized_key] = col

            # 确定要读取的列名
            usecols = []

            # 添加特征列
            for base_col in self.base_features:
                if base_col in temp_column_map:
                    usecols.append(temp_column_map[base_col])

            # 添加标签列
            label_col = None
            for label_name in ['Label', 'label']:
                if label_name in temp_column_map:
                    label_col = temp_column_map[label_name]
                    usecols.append(label_col)
                    break

            if not label_col:
                # 尝试其他方式找标签列
                for col in header_df.columns:
                    if 'label' in col.lower():
                        usecols.append(col)
                        logger.info(f"使用替代标签列: {col}")
                        break

            if not usecols:
                logger.error(f"无法识别需要读取的列")
                return pd.DataFrame()

            logger.info(f"将读取 {len(usecols)} 列: {len(usecols) - 1} 个特征列和 1 个标签列")

            # 分块读取，考虑到文件可能很大，很耗内存
            chunks = None
            try:
                chunks = pd.read_csv(self.data_path, chunksize=10000, usecols=usecols, on_bad_lines='skip')
            except Exception as e:
                logger.warning(f"C 引擎读取失败: {e}，切换 Python 引擎")
                chunks = pd.read_csv(self.data_path, engine='python', chunksize=10000, usecols=usecols,
                                     on_bad_lines='skip')

            chunk_list = []
            for chunk in chunks:
                chunk_list.append(chunk)

            if not chunk_list:
                return pd.DataFrame()

            df = pd.concat(chunk_list, ignore_index=True)
            logger.info(f"文件 {os.path.basename(self.data_path)} 读取完成，shape={df.shape}")

            # 创建列名映射
            self.column_map = self.normalize_column_names(df)

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

        # 1. 移除重复记录
        df_clean = df.drop_duplicates()
        logger.info(f"移除重复记录后剩余 {len(df_clean)} 条记录，df_clean.shape={df_clean.shape}")
        after_dedup = len(df_clean)

        # 获取标签列名
        label_col = None
        for possible_label in ['Label', 'label']:
            possible_col = self.get_actual_column_name(possible_label)
            if possible_col and possible_col in df_clean.columns:
                label_col = possible_col
                break

        if not label_col:
            # 尝试其他方式找标签列
            for col in df_clean.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        # 2. 处理缺失值
        df_clean = self.dropna_in_chunks(df_clean)
        logger.info(f"删除缺失值后剩余 {len(df_clean)} 条记录 (删除了 {after_dedup - len(df_clean)} 条)")

        # 对分类特征使用众数填充
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != label_col:  # 不处理标签列
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # 3. 异常值处理（使用IQR方法）
        for col in numeric_cols:
            if col != label_col:  # 不处理标签列
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 5 * IQR#经实验，取值5，使得有价值的异常仍然保留，即能体现异常，但是又不会因为异常影响运算，且计算较为平稳
                upper_bound = Q3 + 5 * IQR
                # 将异常值限制在边界范围内
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info("数据清洗完成")
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征预处理：独热编码、标准化、归一化、PCA降维"""

        logger.info("开始特征预处理")
        df_processed = df.copy()

        # 1: 处理类别特征（独热编码）
        for base_col in self.categorical_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:
                already_encoded = any(col.startswith(f"{base_col}_") for col in df_processed.columns)
                if already_encoded:
                    logger.info(f"检测到特征 {base_col} 已经完成独热编码，跳过")
                    continue

                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[actual_col]])
                    self.encoders[base_col] = encoder
                else:
                    encoder = self.encoders.get(base_col)
                    if encoder is None:
                        logger.warning(f"找不到特征 {base_col} 的编码器，跳过处理")
                        continue
                    encoded_data = encoder.transform(df_processed[[actual_col]])

                encoded_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_processed.index)

                df_processed = df_processed.drop(actual_col, axis=1)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)

        # 2: 获取标签列
        label_col = None
        for possible_label in ['Label', 'label']:
            actual_col = self.get_actual_column_name(possible_label)
            if actual_col and actual_col in df_processed.columns:
                label_col = actual_col
                break

        if not label_col:
            for col in df_processed.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        # 3: 数值特征归一化
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)

        os.makedirs('models', exist_ok=True)  # 创建模型目录

        if fit:
            self.numeric_feature_order = numeric_cols
            self.minmax_scaler = MinMaxScaler()
            df_processed[numeric_cols] = self.minmax_scaler.fit_transform(df_processed[numeric_cols])

            # 保存 scaler 和特征顺序
            joblib.dump(self.minmax_scaler, 'models/minmax_scaler.pkl')
            with open('models/numeric_feature_order.json', 'w') as f:
                json.dump(self.numeric_feature_order, f)
        else:
            try:
                self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
                with open('models/numeric_feature_order.json', 'r') as f:
                    self.numeric_feature_order = json.load(f)
            except Exception as e:
                raise RuntimeError("验证阶段缺少 scaler 或特征顺序，并且加载失败") from e

            numeric_cols = self.numeric_feature_order
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

        logger.info(f"PCA降维完成，最终特征维数: {result_df.shape[1]}")
        return result_df

    def process_data_pipeline(self, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """完整数据处理流水线"""
        # 1. 加载数据
        df = self.load_data()

        # 2. 数据清洗
        df_clean = self.clean_data(df)

        # 3. 特征预处理
        df_processed = self.preprocess_features(df_clean, fit=train)
        self.last_processed_df = df_processed.copy()

        # 4. 提取特征和标签
        # 获取标签列
        label_col = None
        for possible_label in ['Label', 'label']:
            possible_col = self.get_actual_column_name(possible_label)
            if possible_col and possible_col in df_processed.columns:
                label_col = possible_col
                break

        if not label_col:
            for col in df_processed.columns:
                if 'label' in col.lower():
                    label_col = col
                    logger.info(f"使用替代标签列: {col}")
                    break

        if not label_col:
            logger.error("找不到标签列")
            return np.array([]), np.array([])

        feature_cols = df_processed.columns.tolist()
        feature_cols.remove(label_col)  # 移除标签列

        # 确保特征全是数值类型
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                logger.warning(f"将非数值特征 {col} 转换为数值类型")
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 提取特征和标签
        X = df_processed[feature_cols].values

        # 确保标签是数值类型
        y_data = df_processed[label_col]

        # 检查标签类型并转换
        if pd.api.types.is_object_dtype(y_data):
            logger.info("检测到标签为非数值类型，进行转换")
            # 创建标签映射字典
            unique_labels = np.unique(y_data)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            logger.info(f"标签映射: {label_map}")

            # 转换标签为数值
            y = np.array([label_map[label] for label in y_data], dtype=np.int64)
        else:
            y = y_data.values.astype(np.int64)

        logger.info(f"标签类型: {y.dtype}, 标签唯一值: {np.unique(y)}")

        # 检查特征是否有NaN或无穷值
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("特征数据中包含NaN或无穷值，将其替换为0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def save_preprocessors(self, save_path: str):
        """保存预处理器，包括PCA模型"""
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca_model': self.pca_model,
            'n_components': self.n_components,
            'minmax_scaler': self.minmax_scaler,
            'numeric_feature_order': self.numeric_feature_order
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        logger.info(f"预处理器已保存至 {save_path}")

    def load_preprocessors(self, load_path: str):
        """加载预处理器，包括PCA模型"""
        with open(load_path, 'rb') as f:
            preprocessors = pickle.load(f)

        self.scalers = preprocessors.get('scalers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.pca_model = preprocessors.get('pca_model')
        self.n_components = preprocessors.get('n_components', 20)
        self.minmax_scaler = preprocessors.get('minmax_scaler')
        self.numeric_feature_order = preprocessors.get('numeric_feature_order')

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

    def __init__(self,
                 data_path: str,
                 preprocessor_path: Optional[str] = None,
                 train: bool = True,
                 transform: Optional[Any] = None):
        """初始化数据集"""
        self.transform = transform

        # 初始化处理器
        self.processor = DataProcessor(
            data_path=data_path,
            n_workers=1
        )

        # 如果是训练模式且提供了预处理器路径，在处理后保存预处理器
        if train and preprocessor_path:
            self.features, self.labels = self.processor.process_data_pipeline(train=True)
            logger.info(f"保存预处理器到: {preprocessor_path}")
            # 确保目录存在
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.processor.save_preprocessors(preprocessor_path)
            logger.info("预处理器保存成功")
        # 如果是预测模式且提供了预处理器路径，先加载预处理器再处理数据
        elif not train and preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"预处理器文件不存在: {preprocessor_path}")
            logger.info(f"加载预处理器从: {preprocessor_path}")
            self.processor.load_preprocessors(preprocessor_path)
            logger.info("预处理器加载成功")
            self.features, self.labels = self.processor.process_data_pipeline(train=False)
        else:
            # 无预处理器路径的情况
            self.features, self.labels = self.processor.process_data_pipeline(train=train)
            if train:
                logger.warning("训练模式未提供预处理器保存路径，将无法在预测时使用一致的预处理")

        # 确保数据不为空
        if len(self.features) == 0 or len(self.labels) == 0:
            raise ValueError("处理数据失败，未能生成有效的特征和标签")

        # 检查和打印特征和标签的数据类型
        logger.info(f"特征数据类型: {self.features.dtype}")
        logger.info(f"标签数据类型: {self.labels.dtype}")

        # 明确转换为浮点数(特征)和整数(标签)
        try:
            self.features = np.array(self.features, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

            # 转换为PyTorch张量
            self.features = torch.from_numpy(self.features).float()
            self.labels = torch.from_numpy(self.labels).long().unsqueeze(1)

            logger.info(f"转换后特征形状: {self.features.shape}, 类型: {self.features.dtype}")
            logger.info(f"转换后标签形状: {self.labels.shape}, 类型: {self.labels.dtype}")

        except Exception as e:
            # 如果转换失败，尝试逐行转换
            logger.error(f"标准转换失败: {e}")
            logger.info("尝试逐行转换...")

            # 创建空数组
            float_features = np.zeros((len(self.features), self.features.shape[1]), dtype=np.float32)
            int_labels = np.zeros(len(self.labels), dtype=np.int64)

            # 逐行转换
            for i in range(len(self.features)):
                float_features[i] = [float(val) for val in self.features[i]]

            for i in range(len(self.labels)):
                int_labels[i] = int(self.labels[i])

            # 赋值回原变量并转换为PyTorch张量
            self.features = torch.from_numpy(float_features).float()
            self.labels = torch.from_numpy(int_labels).long().unsqueeze(1)

            logger.info(f"逐行转换后特征形状: {self.features.shape}, 类型: {self.features.dtype}")
            logger.info(f"逐行转换后标签形状: {self.labels.shape}, 类型: {self.labels.dtype}")

    def __len__(self):
        """返回数据集长度"""
        return len(self.features)

    def __getitem__(self, idx):
        """获取单个样本，并调整形状以匹配模型输入需求 (input_size, 1)"""
        x = self.features[idx].unsqueeze(-1)  # 添加最后一个维度，形状变为 (feature_size, 1)
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def get_class_indices(self, selected_classes):
        """
        获取特定类别的样本索引

        参数:
            selected_classes: 需要的类别列表或单个类别

        返回:
            indices: 符合条件的样本索引列表
        """
        if isinstance(selected_classes, (int, np.integer)):
            selected_classes = [selected_classes]

        # 从标签中找出对应类别的索引
        indices = []
        for i in range(len(self.labels)):
            if self.labels[i].item() in selected_classes:
                indices.append(i)

        logger.info(f"找到 {len(indices)} 个属于类别 {selected_classes} 的样本")
        return indices

    def create_class_subset(self, selected_classes):
        """
        创建仅包含特定类别的子数据集

        参数:
            selected_classes: 需要的类别列表或单个类别

        返回:
            subset: 子数据集
        """
        indices = self.get_class_indices(selected_classes)
        return Subset(self, indices)


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
    train_dataset = DDoSDataset(
        data_path="C:\\Users\\17380\\train_dataset.csv",
        preprocessor_path='./outputs\\preprocessor.pkl',
        train=False,
    )
    X_pca, y = train_dataset.processor.get_pca_data()

    # 打印数据形状
    print(f"X_pca shape: {X_pca.shape if X_pca is not None else 'None'}")
    print(f"y shape: {y.shape if y is not None else 'None'}")

    # 打印更多有用信息
    if X_pca is not None and y is not None:
        print(f"X_pca dtype: {X_pca.dtype}")
        print(f"y dtype: {y.dtype}")

        # 打印标签分布情况
        unique_labels, counts = np.unique(y, return_counts=True)
        print("标签分布情况:")
        for label, count in zip(unique_labels, counts):
            print(f"  类别 {label}: {count} 个样本")

        # 打印前几个样本的标签
        print(f"前10个样本标签: {y[:10]}")


if __name__ == "__main__":
    main()
