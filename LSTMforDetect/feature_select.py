import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# 设置图表样式
sns.set_style("whitegrid")


class NetworkTrafficFeatureSelector:
    def __init__(self, csv_path):
        """初始化特征选择器"""
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.feature_importance_results = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("Loading data...")
        # 加载数据
        self.data = pd.read_csv(self.csv_path)
        print(f"Data shape: {self.data.shape}")

        # 检查数据基本信息
        print("\nBasic data information:")
        print(f"Total rows: {len(self.data)}")
        print(f"Total columns: {len(self.data.columns)}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")

        # 分离特征和标签
        self.X = self.data.drop(' Label', axis=1)
        self.y = self.data[' Label']
        self.feature_names = self.X.columns.tolist()

        print(f"Number of features: {len(self.feature_names)}")
        print(f"Label categories: {self.y.unique()}")
        print(f"Label distribution: {self.y.value_counts().to_dict()}")

        # 处理无穷大值和NaN值
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.median())

        # 编码标签
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)

        # 标准化特征（对XGBoost有帮助）
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        print("Data preprocessing completed!")

    def calculate_feature_importance(self):
        """使用多种算法计算特征重要性"""
        print("\nStarting feature importance calculation...")

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )

        # 1. 随机森林
        print("Calculating Random Forest feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

        # 2. Extra Trees
        print("Calculating Extra Trees feature importance...")
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        et_importance = et.feature_importances_
        et_accuracy = accuracy_score(y_test, et.predict(X_test))

        # 3. XGBoost
        print("Calculating XGBoost feature importance...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_importance = xgb_model.feature_importances_
        xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))

        # 保存结果
        self.feature_importance_results = {
            'RandomForest': {'importance': rf_importance, 'accuracy': rf_accuracy},
            'ExtraTrees': {'importance': et_importance, 'accuracy': et_accuracy},
            'XGBoost': {'importance': xgb_importance, 'accuracy': xgb_accuracy}
        }

        # 计算平均重要性
        avg_importance = (rf_importance + et_importance + xgb_importance) / 3
        self.feature_importance_results['Average'] = {'importance': avg_importance}

        print("Feature importance calculation completed!")
        print(
            f"Model accuracies - Random Forest: {rf_accuracy:.4f}, Extra Trees: {et_accuracy:.4f}, XGBoost: {xgb_accuracy:.4f}")

    def create_feature_importance_dataframe(self):
        """创建特征重要性数据框"""
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'RandomForest': self.feature_importance_results['RandomForest']['importance'],
            'ExtraTrees': self.feature_importance_results['ExtraTrees']['importance'],
            'XGBoost': self.feature_importance_results['XGBoost']['importance'],
            'Average': self.feature_importance_results['Average']['importance']
        })

        # 按平均重要性排序
        importance_df = importance_df.sort_values('Average', ascending=False)
        return importance_df

    def visualize_feature_importance(self):
        """可视化特征重要性"""
        print("\nGenerating visualization charts...")

        importance_df = self.create_feature_importance_dataframe()

        # 创建图形
        fig = plt.figure(figsize=(20, 16))

        # 1. 平均特征重要性条形图
        ax1 = plt.subplot(2, 2, 1)
        top_features = importance_df.head(15)  # 显示前15个重要特征
        bars = ax1.barh(range(len(top_features)), top_features['Average'],
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'], fontsize=10)
        ax1.set_xlabel('Feature Importance', fontsize=12)
        ax1.set_title('Top 15 Feature Importance (Average)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)

        # 2. 算法对比热力图
        ax2 = plt.subplot(2, 2, 2)
        heatmap_data = importance_df[['RandomForest', 'ExtraTrees', 'XGBoost']].head(20).T
        heatmap_data.columns = [f'F{i + 1}' for i in range(len(heatmap_data.columns))]
        sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Importance'})
        ax2.set_title('Top 20 Feature Importance Algorithm Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Features (sorted by importance)', fontsize=12)
        ax2.set_ylabel('Algorithm', fontsize=12)

        # 3. 累积重要性曲线
        ax3 = plt.subplot(2, 2, 3)
        cumulative_importance = np.cumsum(importance_df['Average'])
        ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
                 'b-', linewidth=2, marker='o', markersize=3)
        ax3.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Importance')
        ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Importance')
        ax3.set_xlabel('Number of Features', fontsize=12)
        ax3.set_ylabel('Cumulative Importance', fontsize=12)
        ax3.set_title('Cumulative Feature Importance Curve', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 找到80%和90%重要性对应的特征数量
        features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        features_90 = np.argmax(cumulative_importance >= 0.9) + 1
        ax3.axvline(x=features_80, color='r', linestyle=':', alpha=0.7)
        ax3.axvline(x=features_90, color='orange', linestyle=':', alpha=0.7)
        ax3.text(features_80, 0.82, f'{features_80} features', ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax3.text(features_90, 0.92, f'{features_90} features', ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # 4. 特征重要性分布直方图
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(importance_df['Average'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=importance_df['Average'].mean(), color='red', linestyle='--',
                    label=f'Mean: {importance_df["Average"].mean():.4f}')
        ax4.set_xlabel('Feature Importance', fontsize=12)
        ax4.set_ylabel('Number of Features', fontsize=12)
        ax4.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return importance_df, features_80, features_90

    def generate_feature_selection_report(self, importance_df, features_80, features_90):
        """生成特征选择报告"""
        print("\n" + "=" * 60)
        print("FEATURE SELECTION ANALYSIS REPORT")
        print("=" * 60)

        print(f"\n📊 Dataset Basic Information:")
        print(f"   Total features: {len(self.feature_names)}")
        print(f"   Sample count: {len(self.data)}")
        print(f"   Number of classes: {len(np.unique(self.y))}")

        print(f"\n🏆 Model Performance:")
        for model, results in self.feature_importance_results.items():
            if 'accuracy' in results:
                print(f"   {model}: {results['accuracy']:.4f}")

        print(f"\n🔝 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['Feature']:<30} ({row['Average']:.4f})")

        print(f"\n📈 Feature Selection Recommendations:")
        print(f"   Features needed for 80% importance: {features_80}")
        print(f"   Features needed for 90% importance: {features_90}")
        print(f"   Recommended feature count: {features_80}-{features_90}")

        # 推荐特征列表
        recommended_features_80 = importance_df.head(features_80)['Feature'].tolist()
        recommended_features_90 = importance_df.head(features_90)['Feature'].tolist()

        print(f"\n📋 Recommended Feature List (80% importance):")
        for i, feature in enumerate(recommended_features_80, 1):
            print(f"   {i:2d}. {feature}")

        return {
            'top_features_80': recommended_features_80,
            'top_features_90': recommended_features_90,
            'importance_df': importance_df
        }

    def run_feature_selection(self):
        """执行完整的特征选择流程"""
        # 1. 加载和预处理数据
        self.load_and_preprocess_data()

        # 2. 计算特征重要性
        self.calculate_feature_importance()

        # 3. 可视化
        importance_df, features_80, features_90 = self.visualize_feature_importance()

        # 4. 生成报告
        results = self.generate_feature_selection_report(importance_df, features_80, features_90)

        return results


def main():
    """主函数"""
    # 数据文件路径
    csv_path = r"C:\Users\17380\processed_dataset_modified.csv"  # 请根据实际路径修改

    # 创建特征选择器
    selector = NetworkTrafficFeatureSelector(csv_path)

    # 执行特征选择
    results = selector.run_feature_selection()

    print("\n✅ 特征选择完成!")
    print("   可视化图表已显示")
    print("   特征重要性结果已输出")

    return results


if __name__ == "__main__":
    results = main()