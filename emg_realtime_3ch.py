#!/usr/bin/env python3
"""
3通道EMG实时可视化系统
基于WAVELETECH设备的实际数据
支持实时波形显示和RMS计算
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
import time
import serial
import struct
from threading import Thread, Lock
import queue


class WaveletechDataParser:
    """WAVELETECH EMG数据解析器"""

    def __init__(self):
        self.sample_rate = 100.0  # 实际采样率约100Hz
        self.gyro_scale = 0.0012  # rad/s per LSB
        self.acc_scale = 0.0005978  # m/s² per LSB

    def parse_packet(self, data):
        """
        解析24字节数据包
        格式: ch1(2) + ch2(2) + ch3(2) + gr_x(2) + gr_y(2) + gr_z(2) + acc_x(2) + acc_y(2) + acc_z(2) = 18字节
        实际设备可能填充到24字节
        """
        if len(data) < 18:  # 最少需要18字节
            return None

        try:
            # 解析EMG通道数据
            emg_ch1 = struct.unpack('<h', data[0:2])[0]
            emg_ch2 = struct.unpack('<h', data[2:4])[0]
            emg_ch3 = struct.unpack('<h', data[4:6])[0]

            # 解析陀螺仪数据
            gr_x = struct.unpack('<h', data[6:8])[0] * self.gyro_scale
            gr_y = struct.unpack('<h', data[8:10])[0] * self.gyro_scale
            gr_z = struct.unpack('<h', data[10:12])[0] * self.gyro_scale

            # 解析加速度计数据
            acc_x = struct.unpack('<h', data[12:14])[0] * self.acc_scale
            acc_y = struct.unpack('<h', data[14:16])[0] * self.acc_scale
            acc_z = struct.unpack('<h', data[16:18])[0] * self.acc_scale

            return {
                'emg': np.array([emg_ch1, emg_ch2, emg_ch3], dtype=float),
                'gyro': np.array([gr_x, gr_y, gr_z]),
                'acc': np.array([acc_x, acc_y, acc_z])
            }
        except Exception as e:
            print(f"解析错误: {e}")
            return None


class EMGDataSource:
    """EMG数据源类，从串口读取实际数据"""

    def __init__(self, port="/dev/cu.usbserial-0001", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.parser = WaveletechDataParser()
        self.serial_conn = None

    def start(self):
        """开始数据采集"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.is_running = True
            self.thread = Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"✓ 已连接到设备: {self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    def stop(self):
        """停止数据采集"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("串口连接已关闭")

    def _read_loop(self):
        """读取循环"""
        buffer = bytearray()

        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer.extend(data)

                    # 解析数据包
                    while len(buffer) >= 24:
                        packet = bytes(buffer[:24])
                        buffer = buffer[24:]

                        parsed = self.parser.parse_packet(packet)
                        if parsed:
                            try:
                                self.data_queue.put_nowait(parsed)
                            except queue.Full:
                                # 队列满了，丢弃最旧的数据
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(parsed)
                                except queue.Empty:
                                    pass
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"读取数据错误: {e}")
                time.sleep(0.1)

    def get_data(self):
        """获取最新数据"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


class EMGVisualizer:
    """3通道EMG实时可视化器"""

    def __init__(self, data_source, sample_rate=100, window_duration=10.0, rms_window=0.2):
        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_channels = 3
        self.window_duration = window_duration
        self.rms_window = rms_window

        # 计算窗口大小
        self.window_size = int(sample_rate * window_duration)
        self.rms_window_size = int(sample_rate * rms_window)

        # 数据缓冲区
        self.time_buffer = deque(maxlen=self.window_size)
        self.emg_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_channels)]
        self.rms_buffers = [deque(maxlen=100) for _ in range(self.num_channels)]  # 保存最近100个RMS值

        # 数据锁
        self.data_lock = Lock()

        # 设置图表
        self.setup_plot()

    def setup_plot(self):
        """设置matplotlib图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表布局
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('WAVELETECH EMG Real-Time Monitoring System', fontsize=16, fontweight='bold')

        # 使用GridSpec创建布局
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[2, 2, 1], width_ratios=[3, 1],
                     hspace=0.3, wspace=0.3)

        # 创建子图
        self.ax_emg = self.fig.add_subplot(gs[0:2, 0])  # EMG波形
        self.ax_rms = self.fig.add_subplot(gs[0:2, 1])  # RMS柱状图
        self.ax_motion = self.fig.add_subplot(gs[2, :])  # 运动数据

        # 设置EMG波形图
        self.ax_emg.set_title('EMG Signal Waveforms', fontsize=14, fontweight='bold')
        self.ax_emg.set_xlabel('Time (seconds)')
        self.ax_emg.set_ylabel('Amplitude (ADC values)')
        self.ax_emg.grid(True, alpha=0.3)
        self.ax_emg.set_xlim(0, self.window_duration)
        self.ax_emg.set_ylim(-35000, 35000)

        # 设置RMS柱状图
        self.ax_rms.set_title('RMS Indicators (200ms window)', fontsize=14, fontweight='bold')
        self.ax_rms.set_xlabel('Channel')
        self.ax_rms.set_ylabel('RMS Value')
        self.ax_rms.grid(True, alpha=0.3, axis='y')
        self.ax_rms.set_xlim(0, self.num_channels + 1)
        self.ax_rms.set_ylim(0, 5000)

        # 设置运动数据图
        self.ax_motion.set_title('Motion Sensor Data', fontsize=14, fontweight='bold')
        self.ax_motion.set_xlabel('Time (seconds)')
        self.ax_motion.set_ylabel('Value')
        self.ax_motion.grid(True, alpha=0.3)
        self.ax_motion.set_xlim(0, self.window_duration)

        # 创建EMG波形线条
        self.emg_lines = []
        self.emg_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        self.emg_labels = ['Channel 1', 'Channel 2', 'Channel 3']

        for i in range(self.num_channels):
            line, = self.ax_emg.plot([], [], color=self.emg_colors[i], linewidth=1.5,
                                     label=self.emg_labels[i], alpha=0.8)
            self.emg_lines.append(line)

        # 创建RMS柱状图
        channel_positions = np.arange(1, self.num_channels + 1)
        self.rms_bars = self.ax_rms.bar(channel_positions, np.zeros(self.num_channels),
                                       color=self.emg_colors, alpha=0.7,
                                       edgecolor='black', linewidth=1.5)

        # 添加RMS数值标签
        self.rms_texts = []
        for i, pos in enumerate(channel_positions):
            text = self.ax_rms.text(pos, 0, '0', ha='center', va='bottom',
                                   fontsize=10, fontweight='bold')
            self.rms_texts.append(text)

        # 创建运动数据线条
        self.gyro_lines = []
        self.acc_lines = []
        gyro_colors = ['#FF9999', '#99FF99', '#9999FF']
        acc_colors = ['#FFCC99', '#CC99FF', '#99FFCC']

        for i in range(3):
            gyro_line, = self.ax_motion.plot([], [], color=gyro_colors[i], linewidth=1,
                                           label=f'Gyro {["X","Y","Z"][i]}', alpha=0.7)
            acc_line, = self.ax_motion.plot([], [], color=acc_colors[i], linewidth=1,
                                          label=f'Accel {["X","Y","Z"][i]}', alpha=0.7, linestyle='--')
            self.gyro_lines.append(gyro_line)
            self.acc_lines.append(acc_line)

        # 添加图例
        self.ax_emg.legend(loc='upper right', fontsize=10)
        self.ax_motion.legend(loc='upper right', fontsize=8, ncol=2)

        # 设置通道标签
        self.ax_rms.set_xticks(channel_positions)
        self.ax_rms.set_xticklabels([f'CH{i+1}' for i in range(self.num_channels)])

        # 添加信息文本
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10,
                                      transform=self.fig.transFigure)

        # 使用subplots_adjust代替tight_layout避免警告
        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08,
                           hspace=0.3, wspace=0.3)

    def calculate_rms(self, data):
        """计算RMS值"""
        if len(data) == 0:
            return 0.0
        return np.sqrt(np.mean(np.array(data) ** 2))

    def update(self, frame):
        """动画更新函数"""
        # 获取新数据
        new_data = self.data_source.get_data()
        if new_data is None:
            return self.emg_lines + list(self.rms_bars) + self.rms_texts + self.gyro_lines + self.acc_lines

        with self.data_lock:
            # 更新时间
            current_time = time.time()
            if not hasattr(self, 'start_time'):
                self.start_time = current_time

            relative_time = current_time - self.start_time

            # 更新数据缓冲区
            self.time_buffer.append(relative_time)
            emg_data = new_data['emg']
            for ch in range(self.num_channels):
                self.emg_buffers[ch].append(emg_data[ch])

            # 保存运动数据（也用缓冲区，但显示时间窗口更短）
            if not hasattr(self, 'motion_time_buffer'):
                self.motion_time_buffer = deque(maxlen=500)  # 5秒数据
                self.gyro_buffers = [deque(maxlen=500) for _ in range(3)]
                self.acc_buffers = [deque(maxlen=500) for _ in range(3)]

            self.motion_time_buffer.append(relative_time)
            gyro_data = new_data['gyro']
            acc_data = new_data['acc']
            for i in range(3):
                self.gyro_buffers[i].append(gyro_data[i])
                self.acc_buffers[i].append(acc_data[i])

            # 更新EMG波形
            if len(self.time_buffer) > 1:
                time_array = np.array(self.time_buffer)

                # 计算显示时间范围
                if len(time_array) >= self.window_size:
                    x_min = time_array[-self.window_size]
                    x_max = time_array[-1]
                else:
                    x_min = 0
                    x_max = max(self.window_duration, time_array[-1])

                # 更新波形线
                for ch in range(self.num_channels):
                    emg_array = np.array(self.emg_buffers[ch])
                    if len(emg_array) >= self.window_size:
                        emg_display = emg_array[-self.window_size:]
                        time_display = time_array[-self.window_size:]
                    else:
                        emg_display = emg_array
                        time_display = time_array

                    self.emg_lines[ch].set_data(time_display - time_display[0], emg_display)

                # 更新x轴范围
                display_duration = min(self.window_duration, time_array[-1] - time_array[0])
                self.ax_emg.set_xlim(0, display_duration)

            # 更新运动数据
            if hasattr(self, 'motion_time_buffer') and len(self.motion_time_buffer) > 1:
                motion_time_array = np.array(self.motion_time_buffer)
                display_duration = min(5.0, motion_time_array[-1] - motion_time_array[0])

                for i in range(3):
                    gyro_array = np.array(self.gyro_buffers[i])
                    acc_array = np.array(self.acc_buffers[i])

                    if len(gyro_array) >= 500:
                        gyro_display = gyro_array[-500:]
                        acc_display = acc_array[-500:]
                        time_display = motion_time_array[-500:]
                    else:
                        gyro_display = gyro_array
                        acc_display = acc_array
                        time_display = motion_time_array

                    self.gyro_lines[i].set_data(time_display - time_display[0], gyro_display)
                    self.acc_lines[i].set_data(time_display - time_display[0], acc_display)

                self.ax_motion.set_xlim(0, display_duration)

                # 动态调整运动数据y轴范围
                all_gyro = list(gyro_array) if len(gyro_array) > 0 else [0]
                all_acc = list(acc_array) if len(acc_array) > 0 else [0]
                gyro_range = max(abs(min(all_gyro)), abs(max(all_gyro)))
                acc_range = max(abs(min(all_acc)), abs(max(all_acc)))

                self.ax_motion.set_ylim(-max(gyro_range, acc_range) * 1.2,
                                       max(gyro_range, acc_range) * 1.2)

            # 计算并更新RMS
            for ch in range(self.num_channels):
                if len(self.emg_buffers[ch]) >= self.rms_window_size:
                    rms_data = list(self.emg_buffers[ch])[-self.rms_window_size:]
                    rms_value = self.calculate_rms(rms_data)
                    self.rms_buffers[ch].append(rms_value)

                    # 更新柱状图
                    self.rms_bars[ch].set_height(rms_value)

                    # 更新数值标签
                    self.rms_texts[ch].set_position((ch + 1, rms_value))
                    self.rms_texts[ch].set_text(f'{rms_value:.0f}')

                    # 动态调整RMS y轴范围
                    if rms_value > self.ax_rms.get_ylim()[1] * 0.9:
                        self.ax_rms.set_ylim(0, rms_value * 1.2)

            # 更新信息文本
            if len(self.time_buffer) > 1:
                elapsed = relative_time
                packet_count = len(self.time_buffer)
                actual_rate = packet_count / elapsed if elapsed > 0 else 0

                info = f"采样率: {actual_rate:.1f} Hz | 时间: {elapsed:.1f}s | 数据包: {packet_count}"
                self.info_text.set_text(info)

        return self.emg_lines + list(self.rms_bars) + self.rms_texts + self.gyro_lines + self.acc_lines

    def start(self):
        """开始可视化"""
        if not self.data_source.start():
            print("无法启动数据源")
            return

        print("开始EMG实时可视化...")
        print("按Ctrl+C退出程序")

        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=True, cache_frame_data=False
        )

        plt.show()

    def stop(self):
        """停止可视化"""
        self.data_source.stop()


def main():
    """主函数"""
    print("WAVELETECH EMG 实时监测系统")
    print("=" * 50)

    # 创建数据源
    data_source = EMGDataSource()

    # 创建可视化器
    visualizer = EMGVisualizer(
        data_source=data_source,
        sample_rate=100,        # 实际采样率
        window_duration=10.0,   # 显示10秒数据
        rms_window=0.2          # 200ms RMS窗口
    )

    try:
        # 开始可视化
        visualizer.start()
    except KeyboardInterrupt:
        print("\n正在关闭程序...")
        visualizer.stop()
        print("程序已关闭")
    except Exception as e:
        print(f"程序错误: {e}")
        visualizer.stop()


if __name__ == '__main__':
    main()