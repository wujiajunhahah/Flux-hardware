"""
WAVELETECH EMG 传感器数据解析模块
根据图片中的信息解析传感器数据包
"""
import struct
from typing import Dict, List, Optional


class WaveletechParser:
    """解析 WAVELETECH EMG 传感器数据"""
    
    def __init__(self):
        self.sample_rate = 10.0  # 默认采样率 10 Hz
        self.gyro_scale = 0.0012  # rad/s per LSB
        self.acc_scale = 0.0005978  # m/s² per LSB
        
    def parse_packet(self, data: bytes) -> Optional[Dict]:
        """
        解析数据包
        根据图片4的信息：
        - ch1, ch2, ch3: EMG通道数据
        - gr_x, gr_y, gr_z: 陀螺仪数据 (rad/s)
        - acc_x, acc_y, acc_z: 加速度计数据 (m/s²)
        """
        if len(data) < 24:  # 至少需要24字节
            return None
            
        try:
            # 解析EMG通道数据 (假设每个通道2字节，共3个通道)
            emg_ch1 = struct.unpack('<h', data[0:2])[0] if len(data) >= 2 else 0
            emg_ch2 = struct.unpack('<h', data[2:4])[0] if len(data) >= 4 else 0
            emg_ch3 = struct.unpack('<h', data[4:6])[0] if len(data) >= 6 else 0
            
            # 解析陀螺仪数据 (gr_x, gr_y, gr_z)
            gr_x = struct.unpack('<h', data[6:8])[0] * self.gyro_scale if len(data) >= 8 else 0.0
            gr_y = struct.unpack('<h', data[8:10])[0] * self.gyro_scale if len(data) >= 10 else 0.0
            gr_z = struct.unpack('<h', data[10:12])[0] * self.gyro_scale if len(data) >= 12 else 0.0
            
            # 解析加速度计数据 (acc_x, acc_y, acc_z)
            acc_x = struct.unpack('<h', data[12:14])[0] * self.acc_scale if len(data) >= 14 else 0.0
            acc_y = struct.unpack('<h', data[14:16])[0] * self.acc_scale if len(data) >= 16 else 0.0
            acc_z = struct.unpack('<h', data[16:18])[0] * self.acc_scale if len(data) >= 18 else 0.0
            
            return {
                'timestamp': None,  # 将在主程序中添加
                'emg': {
                    'ch1': emg_ch1,
                    'ch2': emg_ch2,
                    'ch3': emg_ch3
                },
                'gyro': {
                    'x': gr_x,
                    'y': gr_y,
                    'z': gr_z
                },
                'acc': {
                    'x': acc_x,
                    'y': acc_y,
                    'z': acc_z
                }
            }
        except Exception as e:
            print(f"解析错误: {e}")
            return None
    
    def parse_hex_string(self, hex_str: str) -> Optional[Dict]:
        """从十六进制字符串解析数据"""
        try:
            # 移除空格和换行符
            hex_str = hex_str.replace(' ', '').replace('\n', '').replace('\r', '')
            # 转换为字节
            data = bytes.fromhex(hex_str)
            return self.parse_packet(data)
        except Exception as e:
            print(f"十六进制解析错误: {e}")
            return None

