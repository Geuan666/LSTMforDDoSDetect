import os
import time
import logging
import pandas as pd
import argparse
from ddos_detector import predict_flow_type  # 导入您现有代码中的检测函数
import traceback
import shutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("realtime_detector.log")]
)
logger = logging.getLogger(__name__)


def check_new_files(folder_path, last_check_time):
    """检查文件夹中是否有新的PCAP文件"""
    new_files = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                file_path = os.path.join(folder_path, file)
                # 使用修改时间而不是创建时间，更可靠
                file_mod_time = os.path.getmtime(file_path)

                # 只处理完全写入的文件（如果文件仍在写入，最后修改时间会非常接近当前时间）
                if file_mod_time > last_check_time and time.time() - file_mod_time > 2:
                    new_files.append(file_path)

        if new_files:
            logger.info(f"发现 {len(new_files)} 个新的PCAP文件: {', '.join(os.path.basename(f) for f in new_files)}")
    except Exception as e:
        logger.error(f"检查新文件时出错: {e}")
        logger.error(traceback.format_exc())

    return new_files


def process_new_file(pcap_file, server_ip, model_path, preprocessor_path, svm_models_dir, output_dir,
                     processed_dir=None):
    """处理新的PCAP文件并保存检测结果"""
    start_time = time.time()

    try:
        # 调用现有的预测函数
        results = predict_flow_type(
            pcap_file=pcap_file,
            server_ip=server_ip,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            svm_models_dir=svm_models_dir
        )

        if not results:
            logger.warning(f"文件 {pcap_file} 没有产生任何检测结果")
            return None

        # 生成输出文件名（使用时间戳和随机数确保唯一性）
        timestamp = int(time.time())
        base_name = os.path.basename(pcap_file).split('.')[0]
        all_results_file = os.path.join(output_dir, f"{base_name}_all_{timestamp}.csv")
        attack_results_file = os.path.join(output_dir, f"{base_name}_attacks_{timestamp}.csv")

        # 保存所有结果
        all_df = pd.DataFrame(results)
        all_df.to_csv(all_results_file, index=False)

        # 筛选出攻击流量（非BENIGN）
        attack_df = all_df[all_df['Predicted Type'] != 'BENIGN']

        if len(attack_df) > 0:
            attack_df.to_csv(attack_results_file, index=False)
            has_attacks = True
        else:
            has_attacks = False
            # 如果没有攻击，记录但不创建空文件
            logger.info(f"文件 {pcap_file} 中未检测到攻击流量")

        # 计算处理时间
        processing_time = time.time() - start_time
        flow_count = len(results)
        flows_per_second = flow_count / processing_time if processing_time > 0 else 0

        # 记录性能统计
        logger.info(f"处理文件: {pcap_file}")
        logger.info(f"总流量数: {flow_count}")
        logger.info(f"攻击流量数: {len(attack_df)}")
        logger.info(f"处理时间: {processing_time:.2f}秒")
        logger.info(f"处理速率: {flows_per_second:.2f}流/秒")
        logger.info(f"所有结果保存至: {all_results_file}")

        # 移动已处理的文件到processed目录（如果指定）
        if processed_dir and os.path.exists(pcap_file):
            try:
                # 确保目标目录存在
                os.makedirs(processed_dir, exist_ok=True)

                # 移动文件
                processed_file = os.path.join(processed_dir, os.path.basename(pcap_file))
                shutil.move(pcap_file, processed_file)
                logger.info(f"已将处理过的文件移动到: {processed_file}")
            except Exception as e:
                logger.warning(f"移动已处理文件时出错: {e}")

        return attack_results_file if has_attacks else None

    except Exception as e:
        logger.error(f"处理文件 {pcap_file} 时出错: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """主函数：实现实时检测循环"""
    parser = argparse.ArgumentParser(description='实时DDoS流量检测')
    parser.add_argument('--pcap_dir', required=True, help='PCAP文件监控目录')
    parser.add_argument('--server_ip', required=True, help='服务器IP地址')
    parser.add_argument('--model', default='./outputs/checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--svm_dir', default='./outputs/svm_models', help='SVM模型目录')
    parser.add_argument('--output_dir', default='./detection_results', help='检测结果输出目录')
    parser.add_argument('--processed_dir', default='./processed_pcaps', help='已处理PCAP文件移动目录')
    parser.add_argument('--interval', type=int, default=10, help='检查间隔（秒）')
    parser.add_argument('--clean_cmd', default='', help='检测到攻击时执行的清洗命令，使用{attack_file}作为占位符')

    args = parser.parse_args()

    # 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    if args.processed_dir:
        os.makedirs(args.processed_dir, exist_ok=True)

    # 验证必要的文件是否存在
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return

    if not os.path.exists(args.preprocessor):
        logger.error(f"预处理器文件不存在: {args.preprocessor}")
        return

    logger.info("启动实时DDoS流量检测服务")
    logger.info(f"监控目录: {args.pcap_dir}")
    logger.info(f"服务器IP: {args.server_ip}")
    logger.info(f"检查间隔: {args.interval}秒")

    # 记录最后检查时间
    last_check_time = time.time()

    # 主循环：不断检查新文件
    try:
        while True:
            # 检查新文件
            new_files = check_new_files(args.pcap_dir, last_check_time)

            # 更新最后检查时间（在处理文件前更新，避免重复处理）
            current_time = time.time()
            last_check_time = current_time

            # 处理每个新文件
            for pcap_file in new_files:
                attack_file = process_new_file(
                    pcap_file=pcap_file,
                    server_ip=args.server_ip,
                    model_path=args.model,
                    preprocessor_path=args.preprocessor,
                    svm_models_dir=args.svm_dir,
                    output_dir=args.output_dir,
                    processed_dir=args.processed_dir
                )

                # 如果检测到攻击并且指定了清洗命令，则执行
                if attack_file and os.path.exists(attack_file) and args.clean_cmd:
                    clean_cmd = args.clean_cmd.replace('{attack_file}', attack_file)
                    logger.info(f"执行清洗命令: {clean_cmd}")
                    try:
                        os.system(clean_cmd)
                    except Exception as e:
                        logger.error(f"执行清洗命令时出错: {e}")

            # 等待下一次检查
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("检测服务被用户中断")
    except Exception as e:
        logger.error(f"检测服务出错: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("检测服务已停止")


if __name__ == "__main__":
    main()