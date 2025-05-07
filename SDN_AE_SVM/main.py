import os
import argparse
import numpy as np
from data import prepare_cicids_dataset, prepare_data_for_training
from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(description="AE-SVM模型训练和测试")
    parser.add_argument('--data_dir', type=str, default="C:\\Users\\17380\\Downloads\\MachineLearningCSV", help="原始数据目录")
    parser.add_argument('--processed_dir', type=str, default="./processed_data", help="处理后数据目录")
    parser.add_argument('--preprocessed_dir', type=str, default="./preprocessed_data", help="预处理后数据目录")
    parser.add_argument('--output_dir', type=str, default="./results", help="结果输出目录")
    parser.add_argument('--hidden_dim', type=int, default=25, help="自编码器隐藏层维度")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=64, help="批大小")
    parser.add_argument('--learning_rate', type=float, default=0.3, help="学习率")
    parser.add_argument('--l2_weight', type=float, default=0.0059, help="L2权重正则化参数")
    parser.add_argument('--sparsity_param', type=float, default=0.002, help="稀疏性参数")
    parser.add_argument('--sparsity_reg', type=float, default=0.007, help="稀疏性正则化参数")
    parser.add_argument('--skip_data_prep', action='store_true', help="跳过数据预处理步骤")
    parser.add_argument('--skip_training', action='store_true', help="跳过训练步骤")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.preprocessed_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 步骤1: 数据处理
    if not args.skip_data_prep:
        print("步骤1: 数据处理...")
        prepare_cicids_dataset(args.data_dir, args.processed_dir)

        print("\n步骤2: 数据预处理...")
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_training(
            train_csv=os.path.join(args.processed_dir, "cicids_train.csv"),
            test_csv=os.path.join(args.processed_dir, "cicids_test.csv"),
            output_folder=args.preprocessed_dir
        )
    else:
        print("跳过数据预处理步骤，直接加载预处理后的数据...")
        from data import load_preprocessed_data
        X_train, y_train, X_test, y_test, scaler = load_preprocessed_data(args.preprocessed_dir)

    # 步骤3: 训练模型
    if not args.skip_training:
        print("\n步骤3: 训练模型...")
        params = {
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'l2_weight': args.l2_weight,
            'sparsity_param': args.sparsity_param,
            'sparsity_reg': args.sparsity_reg,
            'svm_kernel': 'rbf',
            'svm_C': 1.0,
            'svm_gamma': 'scale'
        }
        model, training_time = train_model(args.preprocessed_dir, args.output_dir, params)
    else:
        print("跳过训练步骤...")

    # 步骤4: 测试模型
    print("\n步骤4: 测试模型...")
    input_dim = X_train.shape[1]
    metrics, testing_time = test_model(
        model_folder=os.path.join(args.output_dir, "model"),
        data_folder=args.preprocessed_dir,
        output_folder=args.output_dir,
        input_dim=input_dim
    )

    print("\n总结:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC: {metrics['auc']:.4f}")
    print(f"测试耗时: {testing_time:.2f}秒")


if __name__ == "__main__":
    main()