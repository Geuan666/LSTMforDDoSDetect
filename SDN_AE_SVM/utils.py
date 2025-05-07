import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


def evaluate_model(y_true, y_pred, y_score=None):
    """
    评估模型性能

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 预测概率分数 (用于ROC曲线)

    返回:
        性能指标字典
    """
    # 计算基本指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算ROC AUC (如果提供了分数)
    auc_score = None
    if y_score is not None:
        if y_score.ndim > 1:
            # 如果是多类别概率，取第二类的概率(假设是二分类问题)
            y_score = y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)

    # 构建性能指标字典
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'auc': auc_score
    }

    return metrics


def print_metrics(metrics):
    """打印性能指标"""
    print("模型性能指标:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    if metrics['auc'] is not None:
        print(f"AUC: {metrics['auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"TP: {metrics['true_positive']}")
    print(f"TN: {metrics['true_negative']}")
    print(f"FP: {metrics['false_positive']}")
    print(f"FN: {metrics['false_negative']}")


def plot_roc_curve(y_true, y_score):
    """绘制ROC曲线"""
    if y_score.ndim > 1:
        # 如果是多类别概率，取第二类的概率
        y_score = y_score[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC(AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('./results/roc_curve.png')
    plt.show()


def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_loss'])
    plt.title('evaluation_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('./results/training_history.png')
    plt.show()