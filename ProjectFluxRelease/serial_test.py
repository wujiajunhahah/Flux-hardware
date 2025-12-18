#!/usr/bin/env python3
"""
串口数据测试脚本
用于查看WAVELETECH EMG传感器的原始数据输出
"""

import serial
import serial.tools.list_ports
import time
import struct
from data_parser import WaveletechParser


def list_available_ports():
    """列出所有可用的串口"""
    ports = serial.tools.list_ports.comports()
    print("可用串口列表:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
    return ports


def test_serial_output(port=None, duration=10):
    """测试串口数据输出"""
    # 解析器
    parser = WaveletechParser()

    # 如果没有指定端口，使用第一个可用端口
    if port is None:
        ports = list_available_ports()
        if not ports:
            print("未找到可用串口")
            return
        port = ports[0].device

    print(f"\n连接到串口: {port}")
    print(f"测试时长: {duration}秒")
    print("-" * 60)

    try:
        # 连接串口
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )

        if not ser.is_open:
            ser.open()

        print(f"✓ 成功连接到 {port}")
        print("开始接收数据...")
        print("格式: 时间戳 | CH1 | CH2 | CH3 | 陀螺仪(X,Y,Z) | 加速度(X,Y,Z)")
        print("-" * 100)

        start_time = time.time()
        packet_count = 0
        buffer = bytearray()

        while time.time() - start_time < duration:
            try:
                # 检查是否有数据可读
                if ser.in_waiting > 0:
                    # 读取所有可用数据
                    data = ser.read(ser.in_waiting)
                    buffer.extend(data)

                    # 尝试解析数据包 (假设24字节一个包)
                    while len(buffer) >= 24:
                        packet = bytes(buffer[:24])
                        buffer = buffer[24:]

                        # 解析数据包
                        parsed = parser.parse_packet(packet)
                        if parsed:
                            packet_count += 1
                            current_time = time.time() - start_time

                            # 格式化输出
                            emg = parsed['emg']
                            gyro = parsed['gyro']
                            acc = parsed['acc']

                            print(f"{current_time:6.2f}s | "
                                  f"EMG: {emg['ch1']:6d} {emg['ch2']:6d} {emg['ch3']:6d} | "
                                  f"陀螺仪: {gyro['x']:6.3f} {gyro['y']:6.3f} {gyro['z']:6.3f} | "
                                  f"加速度: {acc['x']:7.3f} {acc['y']:7.3f} {acc['z']:7.3f}")

                            # 每10个包显示一次统计信息
                            if packet_count % 10 == 0:
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    rate = packet_count / elapsed
                                    print(f"[统计] 包数量: {packet_count}, 平均采样率: {rate:.1f} Hz")

                else:
                    time.sleep(0.01)  # 短暂休眠

            except Exception as e:
                print(f"读取错误: {e}")
                time.sleep(0.1)

        # 最终统计
        total_time = time.time() - start_time
        print("-" * 100)
        print(f"测试完成!")
        print(f"总包数量: {packet_count}")
        print(f"总时间: {total_time:.2f}秒")
        if total_time > 0:
            print(f"平均采样率: {packet_count/total_time:.1f} Hz")

        ser.close()
        print(f"串口 {port} 已关闭")

    except Exception as e:
        print(f"连接失败: {e}")


def raw_data_monitor(port=None, duration=30):
    """原始数据监控模式（十六进制输出）"""
    if port is None:
        ports = list_available_ports()
        if not ports:
            print("未找到可用串口")
            return
        port = ports[0].device

    print(f"\n原始数据监控模式")
    print(f"串口: {port}")
    print(f"监控时长: {duration}秒")
    print("-" * 60)

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=1
        )

        start_time = time.time()
        byte_count = 0

        while time.time() - start_time < duration:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                byte_count += len(data)

                # 十六进制输出
                hex_str = data.hex().upper()
                # 每16个字节一组
                for i in range(0, len(hex_str), 32):
                    print(f"[{byte_count-len(data)+i//2:04d}] {hex_str[i:i+32]}")

                print()  # 空行分隔
            else:
                time.sleep(0.01)

        ser.close()
        print(f"总共接收 {byte_count} 字节")

    except Exception as e:
        print(f"监控失败: {e}")


def main():
    """主函数"""
    print("WAVELETECH EMG 传感器数据测试工具")
    print("=" * 50)

    # 列出可用端口
    ports = list_available_ports()

    if not ports:
        print("未找到可用串口，请检查设备连接")
        return

    print(f"\n选择操作模式:")
    print("1. 解析数据显示 (推荐)")
    print("2. 原始数据监控 (十六进制)")
    print("3. 指定串口测试")

    try:
        choice = input("\n请选择 (1-3): ").strip()

        if choice == "1":
            test_serial_output(duration=15)
        elif choice == "2":
            raw_data_monitor(duration=15)
        elif choice == "3":
            port_num = int(input(f"请选择串口编号 (0-{len(ports)-1}): "))
            if 0 <= port_num < len(ports):
                test_serial_output(ports[port_num].device, duration=20)
            else:
                print("无效的串口编号")
        else:
            print("使用默认模式: 解析数据显示")
            test_serial_output(duration=15)

    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序")
    except Exception as e:
        print(f"\n程序错误: {e}")


if __name__ == '__main__':
    main()