import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def prepare_cicids_dataset(data_folder, output_folder):
    """
    处理CICIDS2017的MachineLearningCSV数据，并进行合并和拆分
    """
    all_files = []

    # 查找所有CSV文件
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))

    print(f"找到 {len(all_files)} 个CSV文件")

    # 合并所有文件
    dfs = []
    for file in all_files:
        print(f"处理文件: {file}")
        # 读取CSV文件，处理编码和数据问题
        try:
            df = pd.read_csv(file, encoding='utf-8', low_memory=False)
        except:
            df = pd.read_csv(file, encoding='latin1', low_memory=False)

        # 确保有标签列
        if 'Label' not in df.columns and ' Label' in df.columns:
            df = df.rename(columns={' Label': 'Label'})

        dfs.append(df)

    # 合并所有数据
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后的数据: {combined_df.shape}")

    # 提取DDoS攻击和正常流量
    ddos_df = combined_df[combined_df['Label'].str.contains('DDoS|DoS', case=False, na=False)]
    normal_df = combined_df[combined_df['Label'] == 'BENIGN']

    print(f"DDoS攻击数据数量: {len(ddos_df)}")
    print(f"正常流量数据数量: {len(normal_df)}")

    # 处理过大的数据集 - 采样以平衡数据
    if len(normal_df) > 50000:
        normal_df = normal_df.sample(n=50000, random_state=42)

    if len(ddos_df) > 50000:
        ddos_df = ddos_df.sample(n=50000, random_state=42)

    # 合并采样后的数据
    balanced_df = pd.concat([normal_df, ddos_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据

    # 分割为训练集和测试集
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存处理后的数据
    train_df.to_csv(os.path.join(output_folder, "cicids_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "cicids_test.csv"), index=False)

    print(f"处理完成! 训练集: {train_df.shape}, 测试集: {test_df.shape}")
    print(f"数据已保存到: {output_folder}")

    return train_df, test_df


def preprocess_data(df, is_training=True, scaler=None):
    """
    对数据进行预处理：标签编码、处理缺失值和无穷值、归一化

    参数:
        df: 包含特征和标签的DataFrame
        is_training: 是否为训练数据，决定是否创建新的scaler
        scaler: 已拟合的MinMaxScaler对象（仅在测试时使用）

    返回:
        X_scaled: 预处理后的特征矩阵
        y: 标签数组 (1=DDoS攻击, 0=正常)
        scaler: MinMaxScaler对象(如果是训练数据)
    """
    # 提取标签
    if 'Label' in df.columns:
        labels = df['Label'].copy()
        labels = labels.apply(lambda x: 1 if isinstance(x, str) and ('DDoS' in x or 'DoS' in x) else 0)
    else:
        raise ValueError("数据集中未找到'Label'列")

    # 删除无用列
    columns_to_drop = ['Label']
    for col in df.columns:
        if 'Flow ID' in col or 'Source IP' in col or 'Destination IP' in col or 'Timestamp' in col:
            columns_to_drop.append(col)

    df_features = df.drop(columns_to_drop, axis=1)
    print(f"预处理前的特征数: {df_features.shape[1]}")

    # 编码分类变量
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            print(f"对列进行标签编码: {col}")
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))

    # 处理无穷值和缺失值
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    if df_features.isnull().sum().sum() > 0:
        print("发现缺失值，使用均值填充")
        df_features = df_features.fillna(df_features.mean())

    if is_training:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_features)
        return X_scaled, labels.values, scaler
    else:
        if scaler is None:
            raise ValueError("测试数据必须提供已拟合的scaler")
        X_scaled = scaler.transform(df_features)
        return X_scaled, labels.values


def prepare_data_for_training(train_csv, test_csv, output_folder):
    """
    读取CSV文件，进行预处理并保存为NumPy数组，以便训练

    参数:
        train_csv: 训练数据CSV文件路径
        test_csv: 测试数据CSV文件路径
        output_folder: 输出文件夹路径

    返回:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        scaler: 归一化器
    """
    print("读取和预处理训练数据...")
    train_df = pd.read_csv(train_csv)
    X_train, y_train, scaler = preprocess_data(train_df, is_training=True)

    print("读取和预处理测试数据...")
    test_df = pd.read_csv(test_csv)
    X_test, y_test = preprocess_data(test_df, is_training=False,scaler=scaler)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存预处理后的数据和scaler
    np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(output_folder, 'y_test.npy'), y_test)

    # 保存scaler (使用joblib或pickle)
    import joblib
    joblib.dump(scaler, os.path.join(output_folder, 'scaler.pkl'))

    print(f"预处理后的数据已保存到: {output_folder}")
    print(f"训练特征维度: {X_train.shape}")
    print(f"测试特征维度: {X_test.shape}")

    return X_train, y_train, X_test, y_test, scaler


def load_preprocessed_data(data_folder):
    """
    加载预处理好的数据和scaler

    参数:
        data_folder: 数据文件夹路径

    返回:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        scaler: 归一化器
    """
    X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
    y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
    X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
    y_test = np.load(os.path.join(data_folder, 'y_test.npy'))

    import joblib
    scaler = joblib.load(os.path.join(data_folder, 'scaler.pkl'))

    print(f"加载的训练特征维度: {X_train.shape}")
    print(f"加载的测试特征维度: {X_test.shape}")

    return X_train, y_train, X_test, y_test, scaler


def create_torch_datasets(X_train, y_train, X_test, y_test, batch_size=64):
    """
    创建PyTorch数据集和数据加载器

    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        batch_size: 批处理大小

    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        input_dim: 输入特征维度
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)  # 自编码器: 输入 = 输出
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, X_test_tensor)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    input_dim = X_train.shape[1]

    print(f"创建了PyTorch数据加载器 - 输入维度: {input_dim}, 批处理大小: {batch_size}")

    return train_loader, test_loader, input_dim

# 使用示例
if __name__ == "__main__":
    prepare_cicids_dataset(
        data_folder="C:\\Users\\17380\\Downloads\\MachineLearningCSV\\MachineLearningCVE",
        output_folder="./processed_data"
    )
    # 创建输出文件夹
    output_folder = "./preprocessed_data"
    # 预处理数据并保存
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_training(
        train_csv="./processed_data/cicids_train.csv",
        test_csv="./processed_data/cicids_test.csv",
        output_folder=output_folder
    )
    # 创建PyTorch数据加载器
    train_loader, test_loader, input_dim = create_torch_datasets(
        X_train, y_train, X_test, y_test, batch_size=64
    )