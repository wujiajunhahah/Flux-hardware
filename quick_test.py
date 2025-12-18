#!/usr/bin/env python3
"""
快速串口测试 - 直接连接USB串口设备
"""

import serial
import time
import struct
from data_parser import WaveletechParser


def quick_test():
    """快速测试串口数据"""
    port = "/dev/cu.usbserial-0001"
    parser = WaveletechParser()

    print(f"连接到设备: {port}")
    print("监听数据 10 秒...")
    print("-" * 80)

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1
        )

        print("✓ 串口连接成功")
        print("时间    |  CH1   CH2   CH3   |  陀螺仪(X,Y,Z)     |  加速度(X,Y,Z)")
        print("-" * 80)

        start_time = time.time()
        packet_count = 0
        buffer = bytearray()

        while time.time() - start_time < 10:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)

                # 尝试解析24字节的数据包
                while len(buffer) >= 24:
                    packet = bytes(buffer[:24])
                    buffer = buffer[24:]

                    parsed = parser.parse_packet(packet)
                    if parsed:
                        packet_count += 1
                        t = time.time() - start_time

                        emg = parsed['emg']
                        gyro = parsed['gyro']
                        acc = parsed['acc']

                        print(f"{t:5.1f}s | {emg['ch1']:5d} {emg['ch2']:5d} {emg['ch3']:5d} | "
                              f"{gyro['x']:6.3f} {gyro['y']:6.3f} {gyro['z']:6.3f} | "
                              f"{acc['x']:7.3f} {acc['y']:7.3f} {acc['z']:7.3f}")

                        # 每5个包显示原始数据（调试用）
                        if packet_count <= 5:
                            print(f"  原始: {packet.hex().upper()}")
            else:
                time.sleep(0.01)

        ser.close()
        print("-" * 80)
        print(f"测试完成，接收了 {packet_count} 个数据包")

        if packet_count > 0:
            elapsed = time.time() - start_time
            rate = packet_count / elapsed
            print(f"平均采样率: {rate:.1f} Hz")
        else:
            print("⚠️  没有接收到有效数据包")
            print("可能原因：")
            print("  1. 设备未连接或未工作")
            print("  2. 数据包格式不正确")
            print("  3. 串口参数错误")

    except Exception as e:
        print(f"❌ 连接失败: {e}")


if __name__ == '__main__':
    quick_test()