"""
串口读取模块
用于从 WAVELETECH EMG 传感器读取数据
"""
import serial
import serial.tools.list_ports
import time
from typing import Optional, Callable, List
import threading


class SerialReader:
    """串口数据读取器"""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self.callback: Optional[Callable] = None
        self.thread: Optional[threading.Thread] = None
        
    def list_ports(self) -> List[str]:
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port: Optional[str] = None) -> bool:
        """连接到串口"""
        if port:
            self.port = port
            
        if not self.port:
            ports = self.list_ports()
            if ports:
                self.port = ports[0]  # 使用第一个可用端口
            else:
                print("未找到可用串口")
                return False
        
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"已连接到串口: {self.port}")
            return True
        except Exception as e:
            print(f"连接串口失败: {e}")
            return False
    
    def disconnect(self):
        """断开串口连接"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("已断开串口连接")
    
    def set_callback(self, callback: Callable):
        """设置数据回调函数"""
        self.callback = callback
    
    def start_reading(self):
        """开始读取数据（在后台线程中）"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True
    
    def _read_loop(self):
        """读取循环"""
        buffer = bytearray()
        
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                # 读取可用数据
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer.extend(data)
                    
                    # 尝试查找数据包起始标志（可能需要根据实际协议调整）
                    # 这里假设数据包以特定字节开始，或者按固定长度读取
                    while len(buffer) >= 24:  # 至少24字节
                        packet = bytes(buffer[:24])
                        buffer = buffer[24:]
                        
                        if self.callback:
                            self.callback(packet)
                else:
                    time.sleep(0.01)  # 短暂休眠避免CPU占用过高
                    
            except Exception as e:
                print(f"读取数据错误: {e}")
                time.sleep(0.1)
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.serial_conn is not None and self.serial_conn.is_open

