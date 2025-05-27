#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的训练测试集划分脚本
功能：读取CSV -> 删除指定标签 -> 8:2划分 -> 保存
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_dataset(input_file, output_dir, labels_to_remove=['UDPLag', 'WebDDoS'],
                  test_size=0.2, label_col=' Label'):
    """
    划分训练测试集

    Args:
        input_file: 输入CSV文件路径
        output_dir: 输出目录
        labels_to_remove: 要删除的标签列表
        test_size: 测试集比例
        label_col: 标签列名
    """

    print(f"正在读取文件: {input_file}")
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"原始数据: {df.shape[0]:,} 行, {df.shape[1]} 列")

    # 删除指定标签的数据
    print(f"删除标签: {labels_to_remove}")
    mask = ~df[label_col].isin(labels_to_remove)
    df_filtered = df[mask].reset_index(drop=True)
    print(f"过滤后数据: {df_filtered.shape[0]:,} 行")

    # 显示剩余类别
    print(f"剩余类别: {df_filtered[label_col].unique().tolist()}")

    # 准备特征和标签
    X = df_filtered.drop(columns=[label_col])
    y = df_filtered[label_col]

    # 划分数据集
    print(f"按 {int((1 - test_size) * 100)}:{int(test_size * 100)} 比例划分数据集")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print("使用分层抽样")
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print("使用随机抽样")

    # 重新组合数据
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    print(f"训练集: {train_data.shape[0]:,} 行")
    print(f"测试集: {test_data.shape[0]:,} 行")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    train_file = os.path.join(output_dir, 'train_dataset.csv')
    test_file = os.path.join(output_dir, 'test_dataset.csv')

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"训练集已保存: {train_file}")
    print(f"测试集已保存: {test_file}")
    print("完成！")


if __name__ == "__main__":
    # 设置文件路径
    input_file = r"C:\Users\17380\Desktop\ML-Det-main\Training\sampled_data\sampled_dataset.csv"
    output_dir = r"C:\Users\17380\Desktop\ML-Det-main\Training\final_datasets"

    # 执行划分
    split_dataset(input_file, output_dir)
