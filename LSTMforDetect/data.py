#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import pickle
import warnings
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理类，负责加载、清洗和特征工程
    """

    def __init__(self, data_path: str, n_workers: int = 4, n_components: int = 20):
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

        # 不带空格版本的特征列表
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

        # 需要进行对数转换的特征
        self.log_transform_features_base = [
            'FlowBytes/s', 'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s',
            'FlowDuration', 'PacketLengthVariance'
        ]

        # 类别特征
        self.categorical_features_base = ['Protocol']

    def normalize_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        创建标准化的列名映射，将所有列名无空格版本作为键，原始列名作为值
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

            # 分块读取，考虑到文件可能很大(150000行左右)
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
        """分块处理NaN值，避免内存问题"""
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
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # 将异常值限制在边界范围内
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info("数据清洗完成")
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """特征预处理：对数转换、独热编码、标准化"""
        logger.info("开始特征预处理")

        df_processed = df.copy()

        # 1. 对长尾分布特征进行对数转换: X' = log(1 + X)
        for base_col in self.log_transform_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:
                min_val = df_processed[actual_col].min()
                if min_val < 0:
                    df_processed[actual_col] = df_processed[actual_col] - min_val + 1
                df_processed[actual_col] = np.log1p(df_processed[actual_col])

        # 2. 处理类别特征（独热编码）
        for base_col in self.categorical_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:

                # 检查是否已经进行过独热编码
                already_encoded = any(
                    col.startswith(f"{base_col}_") for col in df_processed.columns
                )
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

        # 3. 标准化数值特征
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()

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

        # 4. PCA降维
        # 获取所有数值特征列（排除标签列）
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if numeric_cols:
            if fit:
                # 训练模式：拟合PCA模型
                logger.info(f"执行PCA降维: 从 {len(numeric_cols)} 维降至 {self.n_components} 维")
                self.pca_model = PCA(n_components=self.n_components)
                pca_result = self.pca_model.fit_transform(df_processed[numeric_cols])
                explained_var = sum(self.pca_model.explained_variance_ratio_) * 100
                logger.info(f"PCA降维后保留信息量: {explained_var:.2f}%")
            else:
                # 预测模式：使用已有PCA模型转换
                 if self.pca_model is None:
                    logger.warning("找不到PCA模型，跳过降维")
                    return df_processed
                 pca_result = self.pca_model.transform(df_processed[numeric_cols])

            # 创建PCA结果DataFrame
            pca_columns = [f'pca_component_{i + 1}' for i in range(self.n_components)]
            pca_df = pd.DataFrame(pca_result, columns=pca_columns, index=df_processed.index)

            # 保留标签列和PCA结果
            if label_col:
                # 如果有标签列，保留标签和PCA结果
                result_df = pd.concat([pca_df, df_processed[[label_col]]], axis=1)
            else:
                # 如果没有标签列，只保留PCA结果
             result_df = pca_df
            logger.info(f"PCA降维完成，降维后特征数: {self.n_components}")
            return result_df

        logger.info("特征预处理完成")
        return df_processed

    def process_data_pipeline(self, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """完整数据处理流水线"""
        # 1. 加载数据
        df = self.load_data()

        # 2. 数据清洗
        df_clean = self.clean_data(df)

        # 3. 特征预处理
        df_processed = self.preprocess_features(df_clean, fit=train)

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
            'n_components': self.n_components
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

        logger.info(f"预处理器已从 {load_path} 加载，PCA维度: {self.n_components}")


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
            n_workers=1  # 单进程处理
        )

        # 如果有预处理器路径且不是训练模式，加载预处理器
        if preprocessor_path and not train:
            self.processor.load_preprocessors(preprocessor_path)

        # 处理数据
        self.features, self.labels = self.processor.process_data_pipeline(train=train)

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


def create_dataloader(dataset: DDoSDataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# 测试用例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化数据处理器
    data_path = "C:\\Users\\17380\\DDoS_dataset.csv"
    processor = DataProcessor(data_path=data_path)

    # 执行数据处理流水线
    X, y = processor.process_data_pipeline(train=True)

    print(f"特征数据形状: {X.shape}")
    if y is not None and len(y) > 0:
        print(f"标签数据形状: {y.shape}")
        print(f"标签数据类型: {y.dtype}")
        print(f"标签唯一值: {np.unique(y)}")

        # 确保标签是数值类型再计算比例
        if np.issubdtype(y.dtype, np.number):
            # 假设攻击标签是非零值
            print(f"攻击样本比例: {np.mean(y != 0):.2%}")
        else:
            print(f"标签不是数值类型，无法计算比例")

    # 测试PyTorch数据集
    try:
        dataset = DDoSDataset(data_path=data_path, train=True)
        print(f"数据集大小: {len(dataset)}")
        x, y = dataset[0]
        print(f"样本特征形状: {x.shape}")  # 应该是 (feature_size, 1)
        print(f"样本标签形状: {y.shape}")

        # 测试数据加载器
        dataloader = create_dataloader(dataset, batch_size=32)
        batch_x, batch_y = next(iter(dataloader))
        print(f"批次特征形状: {batch_x.shape}")  # 应该是 (batch_size, feature_size, 1)
        print(f"批次标签形状: {batch_y.shape}")
    except Exception as e:
        print(f"数据集测试失败: {str(e)}")