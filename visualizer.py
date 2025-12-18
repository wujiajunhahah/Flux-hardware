"""
WAVELETECH EMG 传感器数据可视化程序 - 专业版
"""
import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QLabel, 
                             QGroupBox, QGridLayout, QStatusBar, QFrame)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QBrush
import pyqtgraph as pg
from serial_reader import SerialReader
from data_parser import WaveletechParser

# 设置全局样式
pg.setConfigOptions(antialias=True, useOpenGL=True)


class EMGVisualizer(QMainWindow):
    """EMG传感器数据可视化主窗口 - 专业版"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WAVELETECH EMG 传感器数据可视化系统")
        self.setGeometry(50, 50, 1600, 1000)
        
        # 数据存储
        self.max_points = 1000  # 显示最近1000个数据点
        self.time_data = np.zeros(self.max_points)
        self.emg_data = {
            'ch1': np.zeros(self.max_points),
            'ch2': np.zeros(self.max_points),
            'ch3': np.zeros(self.max_points)
        }
        self.gyro_data = {
            'x': np.zeros(self.max_points),
            'y': np.zeros(self.max_points),
            'z': np.zeros(self.max_points)
        }
        self.acc_data = {
            'x': np.zeros(self.max_points),
            'y': np.zeros(self.max_points),
            'z': np.zeros(self.max_points)
        }
        
        self.data_index = 0
        self.start_time = time.time()
        
        # 初始化组件
        self.serial_reader = SerialReader()
        self.parser = WaveletechParser()
        
        # 设置回调
        self.serial_reader.set_callback(self.on_data_received)
        
        # 创建UI
        self.init_ui()
        self.apply_styles()
        
        # 定时器用于更新图表
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(33)  # 30 FPS
        
    def apply_styles(self):
        """应用全局样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #252525;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4a9eff;
            }
            QPushButton {
                background-color: #2d5aa0;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3d6ab0;
            }
            QPushButton:pressed {
                background-color: #1d4a90;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 2px solid #3a3a3a;
                border-radius: 6px;
                padding: 5px;
                min-width: 200px;
            }
            QComboBox:hover {
                border-color: #4a9eff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #e0e0e0;
                selection-background-color: #2d5aa0;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QStatusBar {
                background-color: #252525;
                color: #a0a0a0;
                border-top: 1px solid #3a3a3a;
            }
        """)
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        central_widget.setLayout(main_layout)
        
        # 状态栏（先创建）
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("未连接")
        
        # 顶部控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 主要内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # 左侧：EMG图表
        emg_group = self.create_emg_chart()
        content_layout.addWidget(emg_group, 2)  # 占2份空间
        
        # 右侧：运动传感器图表
        motion_group = self.create_motion_charts()
        content_layout.addWidget(motion_group, 1)  # 占1份空间
        
        main_layout.addLayout(content_layout, 1)
        
        # 底部：实时数据面板
        stats_panel = self.create_stats_panel()
        main_layout.addWidget(stats_panel)
        
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QGroupBox("🔌 连接控制")
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # 串口选择
        port_label = QLabel("串口:")
        port_label.setMinimumWidth(40)
        layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        self.refresh_ports()
        layout.addWidget(self.port_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.setMinimumWidth(80)
        refresh_btn.clicked.connect(self.refresh_ports)
        layout.addWidget(refresh_btn)
        
        # 连接按钮
        self.connect_btn = QPushButton("🔗 连接")
        self.connect_btn.setMinimumWidth(100)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5aa0;
            }
            QPushButton:hover {
                background-color: #3d6ab0;
            }
        """)
        self.connect_btn.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_btn)
        
        layout.addSpacing(30)
        
        # 采样率显示
        rate_label = QLabel("📊 采样率:")
        rate_label.setMinimumWidth(70)
        layout.addWidget(rate_label)
        
        self.sample_rate_label = QLabel("0.0 Hz")
        self.sample_rate_label.setMinimumWidth(100)
        self.sample_rate_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4a9eff;
                background-color: #2a2a2a;
                padding: 5px 15px;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
            }
        """)
        layout.addWidget(self.sample_rate_label)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_emg_chart(self) -> QGroupBox:
        """创建EMG图表"""
        group = QGroupBox("📈 EMG 肌电信号")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建图表
        self.emg_plot = pg.PlotWidget(background='#1a1a1a')
        self.emg_plot.setTitle("EMG信号波形", color='#e0e0e0', size='14pt')
        self.emg_plot.setLabel('left', '幅度 (ADC值)', color='#a0a0a0', **{'font-size': '11pt'})
        self.emg_plot.setLabel('bottom', '时间 (秒)', color='#a0a0a0', **{'font-size': '11pt'})
        
        # 设置坐标轴样式
        self.emg_plot.getAxis('left').setPen(pg.mkPen(color='#606060', width=1))
        self.emg_plot.getAxis('bottom').setPen(pg.mkPen(color='#606060', width=1))
        self.emg_plot.getAxis('left').setTextPen(pg.mkPen(color='#a0a0a0'))
        self.emg_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#a0a0a0'))
        
        # 设置网格
        self.emg_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 设置Y轴范围
        self.emg_plot.setYRange(-5000, 5000)
        
        # 创建图例
        legend = self.emg_plot.addLegend(offset=(10, 10))
        legend.setBrush(pg.mkBrush(color=(30, 30, 30, 200)))
        legend.setPen(pg.mkPen(color='#606060'))
        
        # EMG曲线 - 使用更美观的颜色和样式
        pen1 = pg.mkPen(color='#ff6b6b', width=2)
        pen2 = pg.mkPen(color='#4ecdc4', width=2)
        pen3 = pg.mkPen(color='#45b7d1', width=2)
        
        self.emg_curves = {
            'ch1': self.emg_plot.plot(self.time_data, self.emg_data['ch1'], 
                                      pen=pen1, name='通道 1', antialias=True),
            'ch2': self.emg_plot.plot(self.time_data, self.emg_data['ch2'], 
                                      pen=pen2, name='通道 2', antialias=True),
            'ch3': self.emg_plot.plot(self.time_data, self.emg_data['ch3'], 
                                      pen=pen3, name='通道 3', antialias=True)
        }
        
        layout.addWidget(self.emg_plot)
        group.setLayout(layout)
        return group
    
    def create_motion_charts(self) -> QWidget:
        """创建运动传感器图表"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 陀螺仪图表
        gyro_group = QGroupBox("🔄 陀螺仪")
        gyro_layout = QVBoxLayout()
        gyro_layout.setContentsMargins(10, 10, 10, 10)
        
        self.gyro_plot = pg.PlotWidget(background='#1a1a1a')
        self.gyro_plot.setTitle("角速度 (rad/s)", color='#e0e0e0', size='12pt')
        self.gyro_plot.setLabel('left', 'rad/s', color='#a0a0a0', **{'font-size': '10pt'})
        self.gyro_plot.setLabel('bottom', '时间 (s)', color='#a0a0a0', **{'font-size': '10pt'})
        
        # 设置坐标轴样式
        self.gyro_plot.getAxis('left').setPen(pg.mkPen(color='#606060', width=1))
        self.gyro_plot.getAxis('bottom').setPen(pg.mkPen(color='#606060', width=1))
        self.gyro_plot.getAxis('left').setTextPen(pg.mkPen(color='#a0a0a0'))
        self.gyro_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#a0a0a0'))
        
        self.gyro_plot.showGrid(x=True, y=True, alpha=0.3)
        self.gyro_plot.setYRange(-10, 10)
        
        legend_gyro = self.gyro_plot.addLegend(offset=(10, 10))
        legend_gyro.setBrush(pg.mkBrush(color=(30, 30, 30, 200)))
        legend_gyro.setPen(pg.mkPen(color='#606060'))
        
        pen_x = pg.mkPen(color='#ff6b6b', width=2)
        pen_y = pg.mkPen(color='#95e1d3', width=2)
        pen_z = pg.mkPen(color='#f38181', width=2)
        
        self.gyro_curves = {
            'x': self.gyro_plot.plot(self.time_data, self.gyro_data['x'], 
                                    pen=pen_x, name='X轴', antialias=True),
            'y': self.gyro_plot.plot(self.time_data, self.gyro_data['y'], 
                                    pen=pen_y, name='Y轴', antialias=True),
            'z': self.gyro_plot.plot(self.time_data, self.gyro_data['z'], 
                                    pen=pen_z, name='Z轴', antialias=True)
        }
        gyro_layout.addWidget(self.gyro_plot)
        gyro_group.setLayout(gyro_layout)
        
        # 加速度计图表
        acc_group = QGroupBox("⚡ 加速度计")
        acc_layout = QVBoxLayout()
        acc_layout.setContentsMargins(10, 10, 10, 10)
        
        self.acc_plot = pg.PlotWidget(background='#1a1a1a')
        self.acc_plot.setTitle("加速度 (m/s²)", color='#e0e0e0', size='12pt')
        self.acc_plot.setLabel('left', 'm/s²', color='#a0a0a0', **{'font-size': '10pt'})
        self.acc_plot.setLabel('bottom', '时间 (s)', color='#a0a0a0', **{'font-size': '10pt'})
        
        # 设置坐标轴样式
        self.acc_plot.getAxis('left').setPen(pg.mkPen(color='#606060', width=1))
        self.acc_plot.getAxis('bottom').setPen(pg.mkPen(color='#606060', width=1))
        self.acc_plot.getAxis('left').setTextPen(pg.mkPen(color='#a0a0a0'))
        self.acc_plot.getAxis('bottom').setTextPen(pg.mkPen(color='#a0a0a0'))
        
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.setYRange(-50, 50)
        
        legend_acc = self.acc_plot.addLegend(offset=(10, 10))
        legend_acc.setBrush(pg.mkBrush(color=(30, 30, 30, 200)))
        legend_acc.setPen(pg.mkPen(color='#606060'))
        
        pen_ax = pg.mkPen(color='#ffa07a', width=2)
        pen_ay = pg.mkPen(color='#98d8c8', width=2)
        pen_az = pg.mkPen(color='#f7dc6f', width=2)
        
        self.acc_curves = {
            'x': self.acc_plot.plot(self.time_data, self.acc_data['x'], 
                                   pen=pen_ax, name='X轴', antialias=True),
            'y': self.acc_plot.plot(self.time_data, self.acc_data['y'], 
                                   pen=pen_ay, name='Y轴', antialias=True),
            'z': self.acc_plot.plot(self.time_data, self.acc_data['z'], 
                                   pen=pen_az, name='Z轴', antialias=True)
        }
        acc_layout.addWidget(self.acc_plot)
        acc_group.setLayout(acc_layout)
        
        layout.addWidget(gyro_group, 1)
        layout.addWidget(acc_group, 1)
        widget.setLayout(layout)
        return widget
    
    def create_stats_panel(self) -> QWidget:
        """创建数据统计面板"""
        panel = QGroupBox("📊 实时数据监控")
        layout = QGridLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题样式
        title_style = """
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #4a9eff;
                padding: 5px;
            }
        """
        
        # 数值样式
        value_style = """
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #e0e0e0;
                background-color: #2a2a2a;
                padding: 8px 15px;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
                min-width: 120px;
            }
        """
        
        # EMG数据
        emg_title = QLabel("EMG 通道")
        emg_title.setStyleSheet(title_style)
        layout.addWidget(emg_title, 0, 0)
        
        self.emg_labels = {
            'ch1': QLabel("CH1: 0")
        }
        self.emg_labels['ch1'].setStyleSheet(value_style.replace('#e0e0e0', '#ff6b6b'))
        layout.addWidget(self.emg_labels['ch1'], 0, 1)
        
        self.emg_labels['ch2'] = QLabel("CH2: 0")
        self.emg_labels['ch2'].setStyleSheet(value_style.replace('#e0e0e0', '#4ecdc4'))
        layout.addWidget(self.emg_labels['ch2'], 0, 2)
        
        self.emg_labels['ch3'] = QLabel("CH3: 0")
        self.emg_labels['ch3'].setStyleSheet(value_style.replace('#e0e0e0', '#45b7d1'))
        layout.addWidget(self.emg_labels['ch3'], 0, 3)
        
        # 陀螺仪数据
        gyro_title = QLabel("陀螺仪")
        gyro_title.setStyleSheet(title_style)
        layout.addWidget(gyro_title, 1, 0)
        
        self.gyro_labels = {
            'x': QLabel("X: 0.000")
        }
        self.gyro_labels['x'].setStyleSheet(value_style.replace('#e0e0e0', '#ff6b6b'))
        layout.addWidget(self.gyro_labels['x'], 1, 1)
        
        self.gyro_labels['y'] = QLabel("Y: 0.000")
        self.gyro_labels['y'].setStyleSheet(value_style.replace('#e0e0e0', '#95e1d3'))
        layout.addWidget(self.gyro_labels['y'], 1, 2)
        
        self.gyro_labels['z'] = QLabel("Z: 0.000")
        self.gyro_labels['z'].setStyleSheet(value_style.replace('#e0e0e0', '#f38181'))
        layout.addWidget(self.gyro_labels['z'], 1, 3)
        
        # 加速度计数据
        acc_title = QLabel("加速度计")
        acc_title.setStyleSheet(title_style)
        layout.addWidget(acc_title, 2, 0)
        
        self.acc_labels = {
            'x': QLabel("X: 0.000")
        }
        self.acc_labels['x'].setStyleSheet(value_style.replace('#e0e0e0', '#ffa07a'))
        layout.addWidget(self.acc_labels['x'], 2, 1)
        
        self.acc_labels['y'] = QLabel("Y: 0.000")
        self.acc_labels['y'].setStyleSheet(value_style.replace('#e0e0e0', '#98d8c8'))
        layout.addWidget(self.acc_labels['y'], 2, 2)
        
        self.acc_labels['z'] = QLabel("Z: 0.000")
        self.acc_labels['z'].setStyleSheet(value_style.replace('#e0e0e0', '#f7dc6f'))
        layout.addWidget(self.acc_labels['z'], 2, 3)
        
        panel.setLayout(layout)
        return panel
    
    def refresh_ports(self):
        """刷新串口列表"""
        self.port_combo.clear()
        ports = self.serial_reader.list_ports()
        self.port_combo.addItems(ports)
        if ports and hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"✓ 找到 {len(ports)} 个可用串口")
    
    def toggle_connection(self):
        """切换连接状态"""
        if self.serial_reader.is_connected():
            self.serial_reader.disconnect()
            self.connect_btn.setText("🔗 连接")
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d5aa0;
                }
                QPushButton:hover {
                    background-color: #3d6ab0;
                }
            """)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("✗ 已断开连接")
        else:
            port = self.port_combo.currentText()
            if port and self.serial_reader.connect(port):
                if self.serial_reader.start_reading():
                    self.connect_btn.setText("🔌 断开")
                    self.connect_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #d32f2f;
                        }
                        QPushButton:hover {
                            background-color: #e53935;
                        }
                    """)
                    if hasattr(self, 'status_bar'):
                        self.status_bar.showMessage(f"✓ 已连接到 {port}")
                    self.start_time = time.time()
                    self.data_index = 0
            else:
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("✗ 连接失败，请检查串口")
    
    def on_data_received(self, data: bytes):
        """接收到数据时的回调"""
        parsed = self.parser.parse_packet(data)
        if parsed:
            current_time = time.time() - self.start_time
            
            # 更新数据数组
            idx = self.data_index % self.max_points
            
            self.time_data[idx] = current_time
            self.emg_data['ch1'][idx] = parsed['emg']['ch1']
            self.emg_data['ch2'][idx] = parsed['emg']['ch2']
            self.emg_data['ch3'][idx] = parsed['emg']['ch3']
            
            self.gyro_data['x'][idx] = parsed['gyro']['x']
            self.gyro_data['y'][idx] = parsed['gyro']['y']
            self.gyro_data['z'][idx] = parsed['gyro']['z']
            
            self.acc_data['x'][idx] = parsed['acc']['x']
            self.acc_data['y'][idx] = parsed['acc']['y']
            self.acc_data['z'][idx] = parsed['acc']['z']
            
            self.data_index += 1
    
    def update_plots(self):
        """更新图表"""
        if self.data_index == 0:
            return
        
        # 计算实际数据范围
        end_idx = min(self.data_index, self.max_points)
        start_idx = max(0, end_idx - self.max_points)
        
        time_window = self.time_data[start_idx:end_idx]
        
        # 更新EMG曲线
        for ch in ['ch1', 'ch2', 'ch3']:
            data_window = self.emg_data[ch][start_idx:end_idx]
            self.emg_curves[ch].setData(time_window, data_window)
        
        # 更新陀螺仪曲线
        for axis in ['x', 'y', 'z']:
            data_window = self.gyro_data[axis][start_idx:end_idx]
            self.gyro_curves[axis].setData(time_window, data_window)
        
        # 更新加速度计曲线
        for axis in ['x', 'y', 'z']:
            data_window = self.acc_data[axis][start_idx:end_idx]
            self.acc_curves[axis].setData(time_window, data_window)
        
        # 更新实时数据标签
        if end_idx > 0:
            last_idx = (end_idx - 1) % self.max_points
            self.emg_labels['ch1'].setText(f"CH1: {self.emg_data['ch1'][last_idx]:.0f}")
            self.emg_labels['ch2'].setText(f"CH2: {self.emg_data['ch2'][last_idx]:.0f}")
            self.emg_labels['ch3'].setText(f"CH3: {self.emg_data['ch3'][last_idx]:.0f}")
            
            self.gyro_labels['x'].setText(f"X: {self.gyro_data['x'][last_idx]:.3f}")
            self.gyro_labels['y'].setText(f"Y: {self.gyro_data['y'][last_idx]:.3f}")
            self.gyro_labels['z'].setText(f"Z: {self.gyro_data['z'][last_idx]:.3f}")
            
            self.acc_labels['x'].setText(f"X: {self.acc_data['x'][last_idx]:.3f}")
            self.acc_labels['y'].setText(f"Y: {self.acc_data['y'][last_idx]:.3f}")
            self.acc_labels['z'].setText(f"Z: {self.acc_data['z'][last_idx]:.3f}")
        
        # 更新采样率
        if self.data_index > 1:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                sample_rate = self.data_index / elapsed
                self.sample_rate_label.setText(f"{sample_rate:.1f} Hz")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.serial_reader.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用图标和名称
    app.setApplicationName("WAVELETECH EMG 可视化系统")
    
    window = EMGVisualizer()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
