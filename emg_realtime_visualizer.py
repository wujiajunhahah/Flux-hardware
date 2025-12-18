#!/usr/bin/env python3
"""
多通道EMG实时可视化系统
支持8通道EMG数据显示，包含实时波形和RMS指示器
采样率: 1000Hz, RMS窗口: 200ms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
import time
from threading import Thread
import queue


class EMGDataSource:
    """EMG数据源类，模拟8通道EMG数据"""

    def __init__(self, sample_rate=1000, num_channels=8):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)

    def start(self):
        """开始数据生成"""
        self.is_running = True
        self.thread = Thread(target=self._generate_data, daemon=True)
        self.thread.start()

    def stop(self):
        """停止数据生成"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)

    def _generate_data(self):
        """生成模拟EMG数据"""
        t = 0
        while self.is_running:
            # 生成8通道模拟EMG数据
            sample = np.zeros(self.num_channels)

            # 为每个通道添加不同特征的信号
            for ch in range(self.num_channels):
                # 基础噪声
                noise = np.random.normal(0, 0.1)

                # 模拟肌肉激活的脉冲信号
                if ch % 2 == 0:  # 偶数通道
                    activation = 0.5 * np.sin(2 * np.pi * 0.5 * t) * (np.random.random() > 0.95)
                else:  # 奇数通道
                    activation = 0.3 * np.sin(2 * np.pi * 0.3 * t + np.pi/4) * (np.random.random() > 0.97)

                # 添加50Hz工频干扰
                powerline = 0.05 * np.sin(2 * np.pi * 50 * t)

                sample[ch] = noise + activation + powerline

            try:
                self.data_queue.put_nowait(sample)
            except queue.Full:
                # 如果队列满了，丢弃最旧的数据
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(sample)
                except queue.Empty:
                    pass

            t += 1.0 / self.sample_rate
            time.sleep(1.0 / self.sample_rate)

    def get_data(self):
        """获取最新数据"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


class EMGVisualizer:
    """8通道EMG实时可视化器"""

    def __init__(self, sample_rate=1000, window_duration=5.0, rms_window=0.2):
        self.sample_rate = sample_rate
        self.num_channels = 8
        self.window_duration = window_duration  # 显示窗口时长(秒)
        self.rms_window = rms_window  # RMS窗口时长(秒)

        # 计算窗口大小
        self.window_size = int(sample_rate * window_duration)
        self.rms_window_size = int(sample_rate * rms_window)

        # 数据缓冲区
        self.time_buffer = deque(maxlen=self.window_size)
        self.emg_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_channels)]
        self.rms_buffers = [deque(maxlen=self.window_size) for _ in range(self.num_channels)]

        # 数据源
        self.data_source = EMGDataSource(sample_rate, self.num_channels)

        # 设置图表
        self.setup_plot()

    def setup_plot(self):
        """设置matplotlib图表"""
        # 创建图表布局
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('8通道EMG实时监测系统', fontsize=16, fontweight='bold')

        # 使用GridSpec创建复杂布局
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[2, 2, 1], width_ratios=[3, 1])

        # 创建子图
        self.ax_emg1 = self.fig.add_subplot(gs[0, 0])  # 通道1-4波形
        self.ax_emg2 = self.fig.add_subplot(gs[1, 0])  # 通道5-8波形
        self.ax_rms = self.fig.add_subplot(gs[:, 1])   # RMS柱状图

        # 设置波形图样式
        for ax, title in zip([self.ax_emg1, self.ax_emg2],
                            ['通道 1-4 波形', '通道 5-8 波形']):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('幅度')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.window_duration)
            ax.set_ylim(-2, 2)

        # 设置RMS柱状图样式
        self.ax_rms.set_title('RMS 指示器 (200ms窗口)', fontsize=12, fontweight='bold')
        self.ax_rms.set_xlabel('通道')
        self.ax_rms.set_ylabel('RMS 值')
        self.ax_rms.grid(True, alpha=0.3, axis='y')
        self.ax_rms.set_xlim(0, self.num_channels + 1)
        self.ax_rms.set_ylim(0, 1)

        # 创建波形线条
        self.emg_lines = []
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colors2 = ['#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

        for i in range(4):  # 通道1-4
            line, = self.ax_emg1.plot([], [], color=colors1[i], linewidth=1,
                                     label=f'CH{i+1}', alpha=0.8)
            self.emg_lines.append(line)

        for i in range(4):  # 通道5-8
            line, = self.ax_emg2.plot([], [], color=colors2[i], linewidth=1,
                                     label=f'CH{i+5}', alpha=0.8)
            self.emg_lines.append(line)

        # 创建RMS柱状图
        channel_positions = np.arange(1, self.num_channels + 1)
        colors = colors1 + colors2
        self.rms_bars = self.ax_rms.bar(channel_positions, np.zeros(self.num_channels),
                                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # 添加数值标签
        self.rms_texts = []
        for i, pos in enumerate(channel_positions):
            text = self.ax_rms.text(pos, 0, '0.000', ha='center', va='bottom',
                                   fontsize=8, fontweight='bold')
            self.rms_texts.append(text)

        # 添加图例
        self.ax_emg1.legend(loc='upper right', ncol=4, fontsize=8)
        self.ax_emg2.legend(loc='upper right', ncol=4, fontsize=8)

        # 设置通道标签
        self.ax_rms.set_xticks(channel_positions)
        self.ax_rms.set_xticklabels([f'CH{i+1}' for i in range(self.num_channels)])

        # 调整布局
        plt.tight_layout()

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
            return self.emg_lines + list(self.rms_bars) + self.rms_texts

        # 更新时间
        current_time = time.time()
        if not hasattr(self, 'start_time'):
            self.start_time = current_time

        relative_time = current_time - self.start_time

        # 更新数据缓冲区
        self.time_buffer.append(relative_time)
        for ch in range(self.num_channels):
            self.emg_buffers[ch].append(new_data[ch])

        # 更新波形
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
            self.ax_emg1.set_xlim(0, min(self.window_duration, time_array[-1] - time_array[0]))
            self.ax_emg2.set_xlim(0, min(self.window_duration, time_array[-1] - time_array[0]))

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
                self.rms_texts[ch].set_text(f'{rms_value:.3f}')

                # 动态调整RMS y轴范围
                if rms_value > self.ax_rms.get_ylim()[1] * 0.9:
                    self.ax_rms.set_ylim(0, rms_value * 1.2)

        return self.emg_lines + list(self.rms_bars) + self.rms_texts

    def start(self):
        """开始可视化"""
        # 启动数据源
        self.data_source.start()

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
    print("启动8通道EMG实时监测系统...")
    print("采样率: 1000Hz")
    print("RMS窗口: 200ms")
    print("按Ctrl+C退出程序")

    # 创建可视化器
    visualizer = EMGVisualizer(
        sample_rate=1000,
        window_duration=5.0,  # 显示5秒数据
        rms_window=0.2         # 200ms RMS窗口
    )

    try:
        # 开始可视化
        visualizer.start()
    except KeyboardInterrupt:
        print("\n正在关闭程序...")
        visualizer.stop()
        print("程序已关闭")


if __name__ == '__main__':
    main()