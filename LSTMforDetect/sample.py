import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict, Counter
from tqdm import tqdm
from IPython.display import display, clear_output
import warnings
import hashlib
import gc

warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)


class CSVSampler:
    def __init__(self, folders, target_columns, label_col=' Label', max_samples_per_class=20000, chunk_size=50000):
        """
        初始化CSV采样器

        Args:
            folders: 要处理的文件夹列表
            target_columns: 用于去重的目标列名列表（去重逻辑基于这些列）
            label_col: 标签列名
            max_samples_per_class: 每个类别最大样本数
            chunk_size: 分块大小
        """
        self.folders = folders
        self.target_columns = target_columns  # 仅用于去重
        self.label_col = label_col
        self.max_samples_per_class = max_samples_per_class
        self.chunk_size = chunk_size

        # 存储每个类别的数据
        self.class_data = defaultdict(list)
        self.class_counts = defaultdict(int)
        self.processed_hashes = set()  # 用于去重

        # 存储所有遇到的列名（用于最终保存）
        self.all_columns = set()

        # 统计信息
        self.total_processed = 0
        self.total_duplicates = 0
        self.file_stats = []

    def clean_label(self, label):
        """清理标签，移除DrDoS_前缀"""
        if isinstance(label, str) and label.startswith('DrDoS_'):
            return label[6:]  # 移除'DrDoS_'前缀
        return label

    def get_row_hash(self, row):
        """生成行的哈希值用于去重（仍然基于目标列）"""
        # 仍然只使用target_columns进行去重，保持原有的去重策略
        try:
            target_data = row[self.target_columns]
            row_str = ''.join(str(val) for val in target_data.values)
            return hashlib.md5(row_str.encode()).hexdigest()
        except KeyError as e:
            # 如果某些target_columns不存在，使用可用的列
            available_target_cols = [col for col in self.target_columns if col in row.index]
            if available_target_cols:
                target_data = row[available_target_cols]
                row_str = ''.join(str(val) for val in target_data.values)
                return hashlib.md5(row_str.encode()).hexdigest()
            else:
                # 如果都不可用，使用所有数值列
                numeric_cols = row.select_dtypes(include=[np.number]).index.tolist()
                if numeric_cols:
                    target_data = row[numeric_cols]
                    row_str = ''.join(str(val) for val in target_data.values)
                    return hashlib.md5(row_str.encode()).hexdigest()
                else:
                    # 最后使用所有列
                    row_str = ''.join(str(val) for val in row.values)
                    return hashlib.md5(row_str.encode()).hexdigest()

    def process_chunk(self, chunk, file_name):
        """处理数据块"""
        chunk_stats = {
            'file': file_name,
            'chunk_size': len(chunk),
            'duplicates': 0,
            'added': 0,
            'skipped_full_classes': 0
        }

        # 记录这个chunk中的所有列名
        self.all_columns.update(chunk.columns.tolist())

        # 清理标签
        if self.label_col in chunk.columns:
            chunk[self.label_col] = chunk[self.label_col].apply(self.clean_label)
        else:
            print(f"警告: 在文件 {file_name} 中找不到标签列 '{self.label_col}'")
            return chunk_stats

        for idx, row in chunk.iterrows():
            # 生成行哈希用于去重（仍然基于target_columns）
            row_hash = self.get_row_hash(row)

            if row_hash in self.processed_hashes:
                chunk_stats['duplicates'] += 1
                self.total_duplicates += 1
                continue

            label = row[self.label_col]

            # 检查该类别是否已经达到最大样本数
            if self.class_counts[label] >= self.max_samples_per_class:
                chunk_stats['skipped_full_classes'] += 1
                continue

            # 添加完整的行数据（所有列），而不仅仅是target_columns
            self.class_data[label].append({
                'data': row.to_dict(),  # 保存完整行数据
                'columns': row.index.tolist()  # 保存列名
            })
            self.class_counts[label] += 1
            self.processed_hashes.add(row_hash)
            chunk_stats['added'] += 1
            self.total_processed += 1

        return chunk_stats

    def process_file(self, file_path):
        """处理单个CSV文件"""
        file_name = os.path.basename(file_path)
        print(f"\n正在处理文件: {file_name}")

        try:
            # 获取文件总行数（用于进度条）
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - 1
            print(f"文件总行数: {total_rows:,}")

            file_stats = {
                'file': file_name,
                'total_rows': total_rows,
                'chunks_processed': 0,
                'total_duplicates': 0,
                'total_added': 0,
                'total_skipped': 0
            }

            # 读取文件头确认列名
            header = pd.read_csv(file_path, nrows=0)
            available_cols = set(header.columns)

            # 检查标签列是否存在
            if self.label_col not in available_cols:
                print(f"警告: 文件 {file_name} 中缺少标签列 '{self.label_col}'")
                return file_stats

            # 检查有多少目标列可用（用于去重）
            available_target_cols = [col for col in self.target_columns if col in available_cols]
            missing_target_cols = [col for col in self.target_columns if col not in available_cols]

            if missing_target_cols:
                print(f"注意: 文件 {file_name} 中缺少以下目标列（用于去重）: {len(missing_target_cols)} 个")

            print(f"文件包含 {len(available_cols)} 列，其中 {len(available_target_cols)} 个目标列可用于去重")

            # 分块读取文件 - 读取所有列，不再限制只读取target_columns
            chunk_iter = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                # 不再指定usecols，读取所有列
                low_memory=False
            )

            # 使用tqdm显示进度
            with tqdm(total=total_rows, desc=f"处理 {file_name}", unit="行") as pbar:
                for chunk in chunk_iter:
                    chunk_stats = self.process_chunk(chunk, file_name)

                    file_stats['chunks_processed'] += 1
                    file_stats['total_duplicates'] += chunk_stats['duplicates']
                    file_stats['total_added'] += chunk_stats['added']
                    file_stats['total_skipped'] += chunk_stats['skipped_full_classes']

                    pbar.update(len(chunk))
                    pbar.set_postfix({
                        '去重': f"{chunk_stats['duplicates']}",
                        '添加': f"{chunk_stats['added']}",
                        '跳过': f"{chunk_stats['skipped_full_classes']}"
                    })

                    # 定期清理内存
                    if file_stats['chunks_processed'] % 10 == 0:
                        gc.collect()

            self.file_stats.append(file_stats)
            return file_stats

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return None

    def process_all_files(self):
        """处理所有文件夹中的CSV文件"""
        all_files = []

        # 收集所有CSV文件
        for folder in self.folders:
            if os.path.exists(folder):
                csv_files = glob.glob(os.path.join(folder, "*.csv"))
                all_files.extend(csv_files)
                print(f"在文件夹 {folder} 中找到 {len(csv_files)} 个CSV文件")
            else:
                print(f"警告: 文件夹 {folder} 不存在")

        print(f"\n总共找到 {len(all_files)} 个CSV文件")

        if not all_files:
            print("没有找到任何CSV文件！")
            return

        # 处理每个文件
        for file_path in all_files:
            self.process_file(file_path)
            self.display_current_stats()

        print(f"\n处理完成！共发现 {len(self.all_columns)} 个不同的列")

    def display_current_stats(self):
        """显示当前统计信息"""
        clear_output(wait=True)

        print("=" * 80)
        print("当前处理统计")
        print("=" * 80)
        print(f"总处理样本数: {self.total_processed:,}")
        print(f"总重复样本数: {self.total_duplicates:,}")
        print(f"当前类别数: {len(self.class_counts)}")
        print(f"发现的总列数: {len(self.all_columns)}")

        # 显示每个类别的样本数
        if self.class_counts:
            print("\n各类别样本数:")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_classes:
                status = "已满" if count >= self.max_samples_per_class else "进行中"
                print(f"  {label}: {count:,} ({status})")

    def save_results(self, output_folder):
        """保存采样结果 - 包含所有原始列"""
        if not self.class_data:
            print("没有数据可保存！")
            return

        print(f"\n开始保存结果到: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

        # 确定最终的列顺序（所有遇到的列的并集）
        all_columns_list = sorted(list(self.all_columns))
        print(f"最终数据集将包含 {len(all_columns_list)} 列")

        # 合并所有类别的数据
        all_rows = []

        print("合并所有类别的数据...")
        for label, data_list in tqdm(self.class_data.items(), desc="合并数据"):
            for item in data_list:
                row_data = item['data']
                # 创建完整行，对于缺失的列用NaN填充
                complete_row = {}
                for col in all_columns_list:
                    complete_row[col] = row_data.get(col, np.nan)
                all_rows.append(complete_row)

        # 创建DataFrame
        print("创建最终DataFrame...")
        final_df = pd.DataFrame(all_rows, columns=all_columns_list)

        # 最终去重检查（基于所有列）
        print("执行最终去重检查...")
        initial_size = len(final_df)
        final_df = final_df.drop_duplicates()
        final_size = len(final_df)
        print(f"最终去重: 移除了 {initial_size - final_size} 个重复行")

        # 保存文件
        output_file = os.path.join(output_folder, 'sampled_dataset.csv')
        print(f"保存文件: {output_file}")
        final_df.to_csv(output_file, index=False)

        # 保存列信息
        columns_info_file = os.path.join(output_folder, 'columns_info.txt')
        with open(columns_info_file, 'w', encoding='utf-8') as f:
            f.write("数据集列信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总列数: {len(all_columns_list)}\n")
            f.write(f"目标列数（用于去重）: {len(self.target_columns)}\n\n")
            f.write("所有列名:\n")
            for i, col in enumerate(all_columns_list, 1):
                is_target = "（用于去重）" if col in self.target_columns else ""
                is_label = "（标签列）" if col == self.label_col else ""
                f.write(f"{i:3d}. {col} {is_target}{is_label}\n")

        # 保存统计信息
        self.save_statistics(output_folder)

        print(f"\n采样完成！")
        print(f"最终数据集大小: {len(final_df):,} 行 x {len(final_df.columns)} 列")
        print(f"保存位置: {output_file}")
        print(f"列信息保存到: {columns_info_file}")

        return final_df

    def save_statistics(self, output_folder):
        """保存统计信息"""
        stats_file = os.path.join(output_folder, 'sampling_statistics.txt')

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("CSV采样统计报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"处理的文件夹:\n")
            for folder in self.folders:
                f.write(f"  - {folder}\n")
            f.write("\n")

            f.write(f"采样参数:\n")
            f.write(f"  - 每类最大样本数: {self.max_samples_per_class:,}\n")
            f.write(f"  - 分块大小: {self.chunk_size:,}\n")
            f.write(f"  - 目标列数（用于去重）: {len(self.target_columns)}\n")
            f.write(f"  - 最终总列数: {len(self.all_columns)}\n")
            f.write("\n")

            f.write(f"处理结果:\n")
            f.write(f"  - 总处理样本数: {self.total_processed:,}\n")
            f.write(f"  - 总重复样本数: {self.total_duplicates:,}\n")
            f.write(f"  - 最终类别数: {len(self.class_counts)}\n")
            f.write("\n")

            f.write("各类别样本数:\n")
            for label, count in sorted(self.class_counts.items()):
                f.write(f"  - {label}: {count:,}\n")
            f.write("\n")

            f.write("去重策略说明:\n")
            f.write("  - 去重基于以下列的组合:\n")
            for col in self.target_columns:
                f.write(f"    * {col}\n")
            f.write("  - 但最终保存所有原始列\n\n")

            if self.file_stats:
                f.write("文件处理详情:\n")
                for stats in self.file_stats:
                    f.write(f"  文件: {stats['file']}\n")
                    f.write(f"    - 总行数: {stats['total_rows']:,}\n")
                    f.write(f"    - 处理块数: {stats['chunks_processed']}\n")
                    f.write(f"    - 重复数: {stats['total_duplicates']:,}\n")
                    f.write(f"    - 添加数: {stats['total_added']:,}\n")
                    f.write(f"    - 跳过数: {stats['total_skipped']:,}\n")
                    f.write("\n")

        print(f"统计信息已保存到: {stats_file}")


# 主执行代码
def main():
    # 定义参数
    folders = [
        r"C:\Users\17380\Desktop\ML-Det-main\Training\01-12",
        r"C:\Users\17380\Desktop\ML-Det-main\Training\03-11"
    ]

    # 这些列仍然用于去重逻辑，但最终会保存所有列
    target_columns = [
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
        ' Inbound'
    ]

    output_folder = r"C:\Users\17380\Desktop\ML-Det-main\Training\sampled_data1"

    # 创建采样器
    sampler = CSVSampler(
        folders=folders,
        target_columns=target_columns,  # 仅用于去重
        label_col=' Label',
        max_samples_per_class=20000,
        chunk_size=50000
    )

    print("开始CSV文件采样处理...")
    print(f"目标文件夹: {folders}")
    print(f"输出文件夹: {output_folder}")
    print(f"每类最大样本数: {sampler.max_samples_per_class:,}")
    print(f"分块大小: {sampler.chunk_size:,}")
    print(f"去重基于的列数: {len(target_columns)} （但将保存所有原始列）")

    # 处理所有文件
    sampler.process_all_files()

    # 保存结果
    final_df = sampler.save_results(output_folder)

    # 显示最终统计
    print("\n" + "=" * 80)
    print("最终统计结果")
    print("=" * 80)

    return final_df


# 运行主程序
if __name__ == "__main__":
    final_dataset = main()