import pandas as pd
from scapy.all import sniff, rdpcap
from scapy.layers.inet import IP, TCP, UDP
from collections import defaultdict
import numpy as np
import pickle
import os
import time
import torch
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'Portmap': 6,
             'SNMP': 7, 'SSDP': 8, 'Syn': 9, 'TFTP': 10, 'UDP': 11, 'UDP-lag': 12}
CLASS_NAMES = list(CLASS_MAP.keys())


# 定义数据结构来存储流的数据包
class Flow:
    def __init__(self):
        self.forward_packets = []  # 上行数据包
        self.backward_packets = []  # 下行数据包
        self.packet_count = 0  # 流中的数据包数量
        self.proto = 6  # TCP=6,UDP=17
        self.flow_name = None

    def add_packet(self, packet, is_upstream):
        if is_upstream:
            self.forward_packets.append(packet)
        else:
            self.backward_packets.append(packet)
        self.packet_count += 1

    def add_proto(self, flow_key):
        self.proto = flow_key[4]
        self.flow_key = flow_key

    def calculate_flow(self):
        """计算流的特征，返回与训练数据完全一致的特征格式"""
        # 以下是您原始代码的计算逻辑

        # 1. 计算基本特征
        if len(self.forward_packets) < len(self.backward_packets):
            inbound = 1
        else:
            inbound = 0

        # 2. 流持续时间
        if self.packet_count > 0:
            start_time = min(pkt.time for pkt in self.forward_packets + self.backward_packets)
            end_time = max(pkt.time for pkt in self.forward_packets + self.backward_packets)
            flow_duration = end_time - start_time
        else:
            flow_duration = 0

        # 3. 计算包长度特征
        packet_lengths = [len(pkt) for pkt in self.forward_packets + self.backward_packets]
        if packet_lengths:
            min_packet_length = min(packet_lengths)
            max_packet_length = max(packet_lengths)
            packet_length_mean = np.mean(packet_lengths)
            packet_length_std = np.std(packet_lengths)
            packet_length_variance = packet_length_std ** 2
            average_packet_size = np.mean(packet_lengths)
        else:
            min_packet_length = 0
            max_packet_length = 0
            packet_length_mean = 0
            packet_length_std = 0
            packet_length_variance = 0
            average_packet_size = 0

        # 6. 流量速率
        if flow_duration > 0:
            flow_packets_per_second = self.packet_count / flow_duration
            flow_bytes_per_second = sum(packet_lengths) / flow_duration if packet_lengths else 0
        else:
            flow_packets_per_second = 0
            flow_bytes_per_second = 0

        # 7. 计算IAT (Inter-Arrival Time)
        def active_times(packets):
            if len(packets) < 2:
                return []
            # 显式转换为 float 类型，避免 Decimal 类型问题
            return [float(packets[i + 1].time - packets[i].time) for i in range(len(packets) - 1)]

        forward_active_times = active_times(self.forward_packets)
        backward_active_times = active_times(self.backward_packets)
        active_times_all = forward_active_times + backward_active_times

        # 计算Flow IAT特征
        if active_times_all:
            # 确保所有值都是 float 类型
            active_times_float = [float(t) for t in active_times_all]
            active_mean = np.mean(active_times_float)
            active_std = np.std(active_times_float)
            active_max = np.max(active_times_float)
            active_min = np.min(active_times_float)
            flow_iat_mean = np.mean(active_times_float)
            flow_iat_std = np.std(active_times_float)
            flow_iat_max = np.max(active_times_float)
            flow_iat_min = np.min(active_times_float)
        else:
            active_mean = 0.0
            active_std = 0.0
            active_max = 0.0
            active_min = 0.0
            flow_iat_mean = 0.0
            flow_iat_std = 0.0
            flow_iat_max = 0.0
            flow_iat_min = 0.0

        # 计算空闲时间特征 (这里简化处理，设置为0)
        idle_mean = 0
        idle_std = 0
        idle_max = 0
        idle_min = 0

        # 计算TCP标志
        fin_flag_count = 0
        syn_flag_count = 0
        rst_flag_count = 0
        psh_flag_count = 0
        ack_flag_count = 0
        urg_flag_count = 0
        cwe_flag_count = 0
        ece_flag_count = 0

        for pkt in self.forward_packets + self.backward_packets:
            if TCP in pkt:
                flags = pkt[TCP].flags
                if flags & 0x01:  # FIN flag
                    fin_flag_count += 1
                if flags & 0x02:  # SYN flag
                    syn_flag_count += 1
                if flags & 0x04:  # RST flag
                    rst_flag_count += 1
                if flags & 0x08:  # PSH flag
                    psh_flag_count += 1
                if flags & 0x10:  # ACK flag
                    ack_flag_count += 1
                if flags & 0x20:  # URG flag
                    urg_flag_count += 1
                if flags & 0x80:  # CWR flag (通常用CWE表示)
                    cwe_flag_count += 1
                if flags & 0x40:  # ECE flag
                    ece_flag_count += 1

        # 前向数据包特征
        total_fwd_packets = len(self.forward_packets)
        fwd_packet_lengths = [len(pkt) for pkt in self.forward_packets]

        if fwd_packet_lengths:
            total_length_of_fwd_packets = sum(fwd_packet_lengths)
            fwd_packet_length_max = max(fwd_packet_lengths)
            fwd_packet_length_min = min(fwd_packet_lengths)
            fwd_packet_length_mean = np.mean(fwd_packet_lengths)
            fwd_packet_length_std = np.std(fwd_packet_lengths)
            avg_fwd_segment_size = fwd_packet_length_mean
        else:
            total_length_of_fwd_packets = 0
            fwd_packet_length_max = 0
            fwd_packet_length_min = 0
            fwd_packet_length_mean = 0
            fwd_packet_length_std = 0
            avg_fwd_segment_size = 0

        # 前向数据包头部长度
        fwd_header_lengths = [pkt[IP].ihl * 4 for pkt in self.forward_packets if IP in pkt]
        if fwd_header_lengths:
            fwd_header_length = sum(fwd_header_lengths)
        else:
            fwd_header_length = 0

        # 前向标志特征
        fwd_psh_flags = sum(1 for pkt in self.forward_packets if TCP in pkt and pkt[TCP].flags & 0x08)
        fwd_urg_flags = sum(1 for pkt in self.forward_packets if TCP in pkt and pkt[TCP].flags & 0x20)

        # 前向数据包速率
        if flow_duration > 0:
            fwd_packets_per_second = total_fwd_packets / flow_duration
        else:
            fwd_packets_per_second = 0

        # 前向IAT特征
        if forward_active_times:
            forward_times_float = [float(t) for t in forward_active_times]
            fwd_iat_total = sum(forward_times_float)
            fwd_iat_mean = np.mean(forward_times_float)
            fwd_iat_std = np.std(forward_times_float)
            fwd_iat_max = np.max(forward_times_float)
            fwd_iat_min = np.min(forward_times_float)
        else:
            fwd_iat_total = 0.0
            fwd_iat_mean = 0.0
            fwd_iat_std = 0.0
            fwd_iat_max = 0.0
            fwd_iat_min = 0.0

        # 前向窗口字节和最小分段大小
        if self.forward_packets and TCP in self.forward_packets[0]:
            init_win_bytes_forward = self.forward_packets[0][TCP].window
            min_seg_size_forward = fwd_packet_length_min
        else:
            init_win_bytes_forward = 0
            min_seg_size_forward = 0

        act_data_pkt_fwd = sum(1 for pkt in self.forward_packets if TCP in pkt and len(pkt[TCP].payload) > 0)

        # 后向数据包特征
        total_bwd_packets = len(self.backward_packets)
        bwd_packet_lengths = [len(pkt) for pkt in self.backward_packets]

        if bwd_packet_lengths:
            total_length_of_bwd_packets = sum(bwd_packet_lengths)
            bwd_packet_length_max = max(bwd_packet_lengths)
            bwd_packet_length_min = min(bwd_packet_lengths)
            bwd_packet_length_mean = np.mean(bwd_packet_lengths)
            bwd_packet_length_std = np.std(bwd_packet_lengths)
            avg_bwd_segment_size = bwd_packet_length_mean
        else:
            total_length_of_bwd_packets = 0
            bwd_packet_length_max = 0
            bwd_packet_length_min = 0
            bwd_packet_length_mean = 0
            bwd_packet_length_std = 0
            avg_bwd_segment_size = 0

        # 后向数据包头部长度
        bwd_header_lengths = [pkt[IP].ihl * 4 for pkt in self.backward_packets if IP in pkt]
        if bwd_header_lengths:
            bwd_header_length = sum(bwd_header_lengths)
        else:
            bwd_header_length = 0

        # 后向标志特征
        bwd_psh_flags = sum(1 for pkt in self.backward_packets if TCP in pkt and pkt[TCP].flags & 0x08)
        bwd_urg_flags = sum(1 for pkt in self.backward_packets if TCP in pkt and pkt[TCP].flags & 0x20)

        # 后向数据包速率
        if flow_duration > 0:
            bwd_packets_per_second = total_bwd_packets / flow_duration
        else:
            bwd_packets_per_second = 0

        # 后向IAT特征
        if backward_active_times:
            backward_times_float = [float(t) for t in backward_active_times]
            bwd_iat_total = sum(backward_times_float)
            bwd_iat_mean = np.mean(backward_times_float)
            bwd_iat_std = np.std(backward_times_float)
            bwd_iat_max = np.max(backward_times_float)
            bwd_iat_min = np.min(backward_times_float)
        else:
            bwd_iat_total = 0.0
            bwd_iat_mean = 0.0
            bwd_iat_std = 0.0
            bwd_iat_max = 0.0
            bwd_iat_min = 0.0

        # 后向窗口字节
        if self.backward_packets and TCP in self.backward_packets[0]:
            init_win_bytes_backward = self.backward_packets[0][TCP].window
        else:
            init_win_bytes_backward = 0

        # 子流特征
        subflow_fwd_packets = total_fwd_packets
        subflow_fwd_bytes = total_length_of_fwd_packets
        subflow_bwd_packets = total_bwd_packets
        subflow_bwd_bytes = total_length_of_bwd_packets

        # 下行/上行比率
        down_up_ratio = total_bwd_packets / total_fwd_packets if total_fwd_packets > 0 else 0

        # 计算流ID
        flow_id = f"{self.flow_key[0]}_{self.flow_key[1]}_{self.flow_key[2]}_{self.flow_key[3]}_{self.flow_key[4]}"

        # 重要：返回与训练时完全一致的特征字典
        # 这里的键名必须与self.base_features完全匹配
        return {
            # 注意：没有空格的特征名称
            'Protocol': self.proto,
            'FlowDuration': flow_duration,
            'TotalFwdPackets': total_fwd_packets,
            'TotalBackwardPackets': total_bwd_packets,
            'FlowBytes/s': flow_bytes_per_second,
            'FlowPackets/s': flow_packets_per_second,
            'FwdPacketLengthMax': fwd_packet_length_max,
            'FwdPacketLengthMin': fwd_packet_length_min,
            'FwdPacketLengthMean': fwd_packet_length_mean,
            'FwdPacketLengthStd': fwd_packet_length_std,
            'BwdPacketLengthMax': bwd_packet_length_max,
            'BwdPacketLengthMin': bwd_packet_length_min,
            'BwdPacketLengthMean': bwd_packet_length_mean,
            'BwdPacketLengthStd': bwd_packet_length_std,
            'PacketLengthVariance': packet_length_variance,
            'FlowIATMin': flow_iat_min,
            'FlowIATMax': flow_iat_max,
            'FlowIATMean': flow_iat_mean,
            'FlowIATStd': flow_iat_std,
            'FwdIATMean': fwd_iat_mean,
            'FwdIATStd': fwd_iat_std,
            'FwdIATMax': fwd_iat_max,
            'FwdIATMin': fwd_iat_min,
            'BwdIATMean': bwd_iat_mean,
            'BwdIATStd': bwd_iat_std,
            'BwdIATMax': bwd_iat_max,
            'BwdIATMin': bwd_iat_min,
            'FwdPSHFlags': fwd_psh_flags,
            'BwdPSHFlags': bwd_psh_flags,
            'FwdURGFlags': fwd_urg_flags,
            'BwdURGFlags': bwd_urg_flags,
            'FwdHeaderLength': fwd_header_length,
            'BwdHeaderLength': bwd_header_length,
            'FwdPackets/s': fwd_packets_per_second,
            'BwdPackets/s': bwd_packets_per_second,
            'Init_Win_bytes_forward': init_win_bytes_forward,
            'Init_Win_bytes_backward': init_win_bytes_backward,
            'min_seg_size_forward': min_seg_size_forward,
            'SubflowFwdBytes': subflow_fwd_bytes,
            'SubflowBwdBytes': subflow_bwd_bytes,
            'AveragePacketSize': average_packet_size,
            'AvgFwdSegmentSize': avg_fwd_segment_size,
            'AvgBwdSegmentSize': avg_bwd_segment_size,
            'ActiveMean': active_mean,
            'ActiveStd': active_std,
            'ActiveMax': active_max,
            'ActiveMin': active_min,
            'IdleMean': idle_mean,
            'IdleStd': idle_std,
            'IdleMax': idle_max,
            'IdleMin': idle_min,
            'Timestamp': time.time(),

            # 以下是额外信息，不用于模型预测
            'Flow ID': flow_id
        }


# 将数据包按流分类
def classify_packets(packet, flows, server_ip):
    if IP in packet:
        # 对于TCP还是UDP，进行属性赋值，防止之后处理流程不同
        if TCP in packet:
            proto = 6
            flow_key = (packet[IP].src, packet[IP].dst, packet[TCP].sport, packet[TCP].dport, proto)
        elif UDP in packet:
            proto = 17
            flow_key = (packet[IP].src, packet[IP].dst, packet[UDP].sport, packet[UDP].dport, proto)
        else:
            return

        # 判断数据包方向,上行还是下行
        if packet[IP].src == server_ip:
            is_upstream = False
        elif packet[IP].dst == server_ip:
            is_upstream = True
        else:
            is_upstream = True

        # 查找流是否已存在
        if flow_key in flows:
            flows[flow_key].add_packet(packet, is_upstream)
            flows[flow_key].add_proto(flow_key)
        else:
            new_flow = Flow()
            new_flow.add_packet(packet, is_upstream)
            new_flow.add_proto(flow_key)
            flows[flow_key] = new_flow


# 分析PCAP文件，提取流特征
def analyze_pcap(pcap_file, server_ip):
    flows = defaultdict(Flow)

    # 只捕获ip数据包，其余的都忽略
    try:
        packets = rdpcap(pcap_file)
        logger.info(f"加载了 {len(packets)} 个数据包")
    except Exception as e:
        logger.error(f"读取PCAP文件时出错: {e}")
        return []

    for packet in packets:
        classify_packets(packet, flows, server_ip)

    flow_features = []
    # 对每个流进行特征计算
    for flow_key, flow in flows.items():
        if flow.packet_count >= 2:  # 忽略只有一个包的流
            flow_feature = flow.calculate_flow()
            flow_features.append(flow_feature)

    logger.info(f"从PCAP文件提取了 {len(flow_features)} 个流特征")
    return flow_features


# 预处理特征用于模型输入
def preprocess_flow_features(flow_features, preprocessor_path):
    """
    预处理提取的流量特征，使其与模型训练数据格式一致

    参数:
        flow_features: 提取的流量特征列表
        preprocessor_path: 预处理器保存路径

    返回:
        预处理后的特征，可直接输入模型
    """
    logger.info("开始预处理流特征...")

    # 转换为DataFrame
    df = pd.DataFrame(flow_features)

    # 创建无空格列名到有空格列名的映射
    column_mapping = {
        'Protocol': 'Protocol',  # 这个不变
        'FlowDuration': 'Flow Duration',
        'TotalFwdPackets': 'Total Fwd Packets',
        'TotalBackwardPackets': 'Total Backward Packets',
        'FlowBytes/s': 'Flow Bytes/s',
        'FlowPackets/s': 'Flow Packets/s',
        'FwdPacketLengthMax': 'Fwd Packet Length Max',
        'FwdPacketLengthMin': 'Fwd Packet Length Min',
        'FwdPacketLengthMean': 'Fwd Packet Length Mean',
        'FwdPacketLengthStd': 'Fwd Packet Length Std',
        'BwdPacketLengthMax': 'Bwd Packet Length Max',
        'BwdPacketLengthMin': 'Bwd Packet Length Min',
        'BwdPacketLengthMean': 'Bwd Packet Length Mean',
        'BwdPacketLengthStd': 'Bwd Packet Length Std',
        'PacketLengthVariance': 'Packet Length Variance',
        'FlowIATMin': 'Flow IAT Min',
        'FlowIATMax': 'Flow IAT Max',
        'FlowIATMean': 'Flow IAT Mean',
        'FlowIATStd': 'Flow IAT Std',
        'FwdIATMean': 'Fwd IAT Mean',
        'FwdIATStd': 'Fwd IAT Std',
        'FwdIATMax': 'Fwd IAT Max',
        'FwdIATMin': 'Fwd IAT Min',
        'BwdIATMean': 'Bwd IAT Mean',
        'BwdIATStd': 'Bwd IAT Std',
        'BwdIATMax': 'Bwd IAT Max',
        'BwdIATMin': 'Bwd IAT Min',
        'FwdPSHFlags': 'Fwd PSH Flags',
        'BwdPSHFlags': 'Bwd PSH Flags',
        'FwdURGFlags': 'Fwd URG Flags',
        'BwdURGFlags': 'Bwd URG Flags',
        'FwdHeaderLength': 'Fwd Header Length',
        'BwdHeaderLength': 'Bwd Header Length',
        'FwdPackets/s': 'Fwd Packets/s',
        'BwdPackets/s': 'Bwd Packets/s',
        'Init_Win_bytes_forward': 'Init_Win_bytes_forward',  # 这些可能在原始数据中没有空格
        'Init_Win_bytes_backward': 'Init_Win_bytes_backward',
        'min_seg_size_forward': 'min_seg_size_forward',
        'SubflowFwdBytes': 'Subflow Fwd Bytes',
        'SubflowBwdBytes': 'Subflow Bwd Bytes',
        'AveragePacketSize': 'Average Packet Size',
        'AvgFwdSegmentSize': 'Avg Fwd Segment Size',
        'AvgBwdSegmentSize': 'Avg Bwd Segment Size',
        'ActiveMean': 'Active Mean',
        'ActiveStd': 'Active Std',
        'ActiveMax': 'Active Max',
        'ActiveMin': 'Active Min',
        'IdleMean': 'Idle Mean',
        'IdleStd': 'Idle Std',
        'IdleMax': 'Idle Max',
        'IdleMin': 'Idle Min',
        'Timestamp': 'Timestamp'  # 这个不变
    }

    # 重命名DataFrame的列名
    df.rename(columns=column_mapping, inplace=True)

    # 加载预处理器
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)

        # 获取预处理器
        scalers = preprocessors.get('scalers', {})
        encoders = preprocessors.get('encoders', {})
        pca_model = preprocessors.get('pca_model')
        minmax_scaler = preprocessors.get('minmax_scaler')
        numeric_feature_order = preprocessors.get('numeric_feature_order')

        logger.info(f"已加载预处理器，PCA维度: {pca_model.n_components if pca_model else 'Unknown'}")
    except Exception as e:
        logger.error(f"加载预处理器时出错: {e}")
        return None

    # 准备特征 - 必须与训练时完全一致
    # 1. 检查base_features中的特征是否都存在
    base_features = [
        'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Packet Length Variance', 'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean',
        'Flow IAT Std', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward', 'min_seg_size_forward', 'Subflow Fwd Bytes',
        'Subflow Bwd Bytes', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Active Mean', 'Active Min', 'Active Max', 'Active Std',
        'Idle Mean', 'Idle Min', 'Idle Max', 'Idle Std', 'Timestamp',
    ]

    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        logger.warning(f"缺少以下特征: {missing_features}")
        for feature in missing_features:
            df[feature] = 0  # 对缺失特征填0

    # 2. 处理类别特征（独热编码）
    categorical_features = ['Protocol']
    for feature in categorical_features:
        if feature in df.columns and feature in encoders:
            encoder = encoders[feature]
            # 确保数据类型正确
            df[feature] = df[feature].astype(int)
            encoded_data = encoder.transform(df[[feature]])
            encoded_cols = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
            df = df.drop(feature, axis=1)
            df = pd.concat([df, encoded_df], axis=1)

    # 3. 确保所有数值特征都是数值类型
    numeric_cols = [col for col in df.columns if col in numeric_feature_order]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 替换NaN和inf
        df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    # 4. 应用归一化
    if numeric_feature_order and minmax_scaler:
        # 确保列顺序一致
        df_numeric = df[numeric_feature_order]
        numeric_data = minmax_scaler.transform(df_numeric)
    else:
        logger.error("无法应用归一化，特征顺序或归一化器缺失")
        return None

    # 5. 应用PCA降维
    if pca_model:
        pca_result = pca_model.transform(numeric_data)
        logger.info(f"PCA降维后特征形状: {pca_result.shape}")
    else:
        logger.error("无法应用PCA降维，PCA模型缺失")
        return None

    # 6. 转换为LSTM模型需要的输入格式 [batch_size, seq_len=25, input_size=1]
    # 模型期望的输入形状是 [batch_size, 25, 1]
    input_tensor = torch.FloatTensor(pca_result).unsqueeze(-1)
    logger.info(f"准备好的模型输入形状: {input_tensor.shape}")

    return input_tensor


# LSTM模型
class BiLSTMDetector(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=13, dropout_rate=0.5):
        super(BiLSTMDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.relu = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # LSTM前向传播
        lstm_out, (final_hidden, _) = self.lstm(x)

        # 重新整形最终隐藏状态以提取最后一层
        final_hidden = final_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        final_forward = final_hidden[-1, 0, :, :]  # 最后一层前向LSTM的隐藏状态
        final_backward = final_hidden[-1, 1, :, :]  # 最后一层后向LSTM的隐藏状态

        # 连接前向和后向状态
        combined = torch.cat((final_forward, final_backward), dim=1)  # [batch_size, hidden_size*2]

        # 应用批归一化
        combined = self.batch_norm(combined)

        # 应用dropout进行正则化
        combined = self.dropout(combined)

        # 通过第一个全连接层
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)

        # 通过第二个全连接层得到最终输出
        output = self.fc2(x)  # [batch_size, num_classes]

        return output


# SVMModel类的实现
class SVMModel:
    """SVM分类器模型类"""

    def __init__(self, class1, class2, kernel='rbf', C=1.0, gamma='scale', probability=True):
        """
        初始化SVM模型

        参数:
            class1: 第一个类别的索引
            class2: 第二个类别的索引
            kernel: 核函数类型
            C: 正则化参数
            gamma: 核系数
            probability: 是否启用概率估计
        """
        self.class1 = class1
        self.class2 = class2
        from sklearn.svm import SVC
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            class_weight='balanced'
        )

    def fit(self, X, y):
        """
        训练SVM模型

        参数:
            X: 特征矩阵
            y: 标签向量（二进制，0表示class1，1表示class2）
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        预测样本类别

        参数:
            X: 特征矩阵

        返回:
            预测的类别: 返回原始类别索引（class1或class2）
        """
        binary_pred = self.model.predict(X)
        # 将二元预测转换回原始类别
        return np.where(binary_pred == 0, self.class1, self.class2)

    def save(self, path):
        """
        保存模型

        参数:
            path: 保存路径
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        加载模型

        参数:
            path: 模型路径

        返回:
            加载的SVM模型实例
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


# SVMCascadeModel类的实现
class SVMCascadeModel:
    """
    SVM级联模型：结合BiLSTM基础分类器和SVM二分类器
    """

    def __init__(self, base_model, confusion_pairs=None, confidence_threshold=0.95):
        """
        初始化级联模型
        参数:
            base_model: 基础BiLSTM模型
            confusion_pairs: 混淆类别对列表，如 [(11,12), (5,7), (2,8), (9,10)]
            confidence_threshold: 不触发二级分类器的置信度阈值
        """
        self.base_model = base_model
        self.confusion_pairs = confusion_pairs or [(11, 12), (5, 7), (2, 8), (9, 10)]
        self.confidence_threshold = confidence_threshold
        self.svm_models = {}

    def load_svm_models(self, model_dir):
        """
        加载SVM二分类器
        参数:
            model_dir: SVM模型目录
        """
        for class1, class2 in self.confusion_pairs:
            model_path = os.path.join(model_dir, f"svm_model_{class1}_{class2}.pkl")
            if os.path.exists(model_path):
                self.svm_models[(class1, class2)] = SVMModel.load(model_path)
                logger.info(f"已加载SVM分类器: {class1} vs {class2}")
            else:
                logger.warning(f"找不到SVM分类器: {class1} vs {class2}")

    def predict(self, inputs, device=None):
        """
        使用级联模型预测

        参数:
            inputs: BiLSTM模型的输入数据，形状为[batch_size, feature_dim, 1]
            device: 计算设备

        返回:
            final_pred: 最终预测标签
            base_pred: 基础模型预测
            probs: 预测概率
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基础BiLSTM模型预测
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(inputs.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            base_pred = outputs.max(1)[1].cpu().numpy()
            confidence = probs.max(1)[0].cpu().numpy()

        # 如果没有加载SVM分类器，直接返回基础预测
        if not self.svm_models:
            return base_pred, base_pred, probs.cpu().numpy()

        # 最终预测结果
        final_pred = base_pred.copy()

        # 提取特征（从输入的倒数第二维）
        pca_features = inputs.squeeze(-1).cpu().numpy()

        # 对每个样本应用SVM二次分类
        for i, (pred, conf) in enumerate(zip(base_pred, confidence)):
            # 检查是否是混淆类别对
            for class1, class2 in self.confusion_pairs:
                if pred in [class1, class2] and conf < self.confidence_threshold:
                    # 找到对应的SVM分类器
                    svm_model = self.svm_models.get((class1, class2))
                    if svm_model is None:
                        continue

                    # 获取当前样本的特征
                    feature = pca_features[i:i + 1]

                    # 预测并更新结果
                    final_pred[i] = svm_model.predict(feature)[0]
                    break

        return final_pred, base_pred, probs.cpu().numpy()


# SVM级联模型加载函数（如果您的项目使用了级联模型）
def load_cascade_model(lstm_model_path, svm_models_dir, confusion_pairs=None):
    """加载LSTM模型和SVM级联分类器"""
    if confusion_pairs is None:
        confusion_pairs = [(11, 12), (5, 7), (2, 8), (9, 10)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载LSTM模型
    model = BiLSTMDetector(input_size=1, hidden_size=128, num_layers=2, num_classes=13)
    checkpoint = torch.load(lstm_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 直接使用我们上面实现的SVMCascadeModel类
    cascade_model = SVMCascadeModel(
        base_model=model,
        confusion_pairs=confusion_pairs,
        confidence_threshold=0.95
    )

    # 加载SVM模型
    cascade_model.load_svm_models(svm_models_dir)

    return cascade_model


# 完整的预测流程
def predict_flow_type(pcap_file, server_ip, model_path, preprocessor_path, svm_models_dir=None):
    """
    完整的流量分类预测流程

    参数:
        pcap_file: PCAP文件路径
        server_ip: 服务器IP地址，用于确定流方向
        model_path: 模型路径
        preprocessor_path: 预处理器路径
        svm_models_dir: SVM模型目录

    返回:
        predictions: 预测结果列表
    """
    # 1. 提取特征
    logger.info(f"开始从 {pcap_file} 提取流特征...")
    flow_features = analyze_pcap(pcap_file, server_ip)

    if not flow_features:
        logger.error("未能提取任何流特征")
        return []

    # 2. 预处理特征
    logger.info("预处理流特征...")
    input_tensor = preprocess_flow_features(flow_features, preprocessor_path)

    if input_tensor is None:
        logger.error("特征预处理失败")
        return []

    # 3. 加载模型
    logger.info(f"加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 如果提供了SVM模型目录，使用级联模型
    if svm_models_dir:
        try:
            model = load_cascade_model(model_path, svm_models_dir)
            use_cascade = True
        except Exception as e:
            logger.error(f"加载级联模型失败: {e}，回退到基本LSTM模型")
            model = BiLSTMDetector(input_size=1, hidden_size=64, num_layers=2, num_classes=13)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            use_cascade = False
    else:
        # 只使用LSTM模型
        model = BiLSTMDetector(input_size=1, hidden_size=64, num_layers=2, num_classes=13)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        use_cascade = False

    # 4. 进行预测
    logger.info("进行预测...")
    with torch.no_grad():
        inputs = input_tensor.to(device)

        if use_cascade:
            final_pred, base_pred, probs = model.predict(inputs, device)
            probabilities = probs
            predicted = final_pred
        else:
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()

    # 5. 整理结果
    results = []
    for i, (pred, prob) in enumerate(zip(predicted, probabilities)):
        flow_id = flow_features[i].get('Flow ID', f"flow_{i}")
        pred_class = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else f"Unknown-{pred}"

        result = {
            'Flow ID': flow_id,
            'Source IP': flow_features[i].get('Source IP', ''),
            'Destination IP': flow_features[i].get('Destination IP', ''),
            'Source Port': flow_features[i].get('Source Port', ''),
            'Destination Port': flow_features[i].get('Destination Port', ''),
            'Protocol': flow_features[i].get('Protocol', ''),
            'Predicted Type': pred_class,
            'Confidence': prob[pred] if isinstance(prob, np.ndarray) else 0.0
        }
        results.append(result)

    logger.info(f"预测完成，共 {len(results)} 个结果")

    # 统计预测结果
    pred_counts = {}
    for result in results:
        pred_type = result['Predicted Type']
        pred_counts[pred_type] = pred_counts.get(pred_type, 0) + 1

    logger.info(f"预测类型统计: {pred_counts}")

    return results


# 主函数示例
def main():
    import argparse

    parser = argparse.ArgumentParser(description='DDoS流量分类预测')
    parser.add_argument('--pcap', required=True, help='PCAP文件路径')
    parser.add_argument('--server_ip', required=True, help='服务器IP地址')
    parser.add_argument('--model', default='./outputs/checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--svm_dir', default='./outputs/svm_models', help='SVM模型目录（如果使用级联模型）')
    parser.add_argument('--output', default='predictions.csv', help='输出CSV文件路径')

    args = parser.parse_args()

    # 执行预测
    results = predict_flow_type(
        pcap_file=args.pcap,
        server_ip=args.server_ip,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        svm_models_dir=args.svm_dir
    )

    # 保存结果到CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"预测结果已保存到 {args.output}")


if __name__ == "__main__":
    main()