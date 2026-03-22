#!/usr/bin/env python3
"""
EMG 信号诊断工具
=========================================================
用途: 验证 ADC 转换公式、信号范围、ZC/SSC 阈值是否合理。

步骤:
  1. 放松手臂 10 秒 (baseline)
  2. 用力握拳 10 秒 (contraction)
  3. 自动对比并输出诊断报告

用法:
  python tools/emg_diagnostic.py --port /dev/tty.usbserial-0001
  python tools/emg_diagnostic.py --port /dev/cu.usbserial-0001
=========================================================
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import remove_dc
from src.stream import SerialEMGStream, CHANNEL_COUNT, _decode_signed24


def collect_phase(stream: SerialEMGStream, duration_sec: float, label: str) -> np.ndarray:
    """采集一段数据，返回 (N, CHANNEL_COUNT) 数组。"""
    print(f"\n>>> {label} ({duration_sec}s) — 现在开始！")
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_sec:
        batch = stream.consume_samples(max_items=256)
        for s in batch:
            samples.append(s.values)
        time.sleep(0.01)
    arr = np.array(samples) if samples else np.zeros((0, CHANNEL_COUNT))
    print(f"    采集到 {len(arr)} 个样本 ({len(arr)/max(duration_sec,1):.0f} Hz)")
    return arr


def analyze(data: np.ndarray, label: str) -> dict:
    """计算基本统计量。"""
    if data.size == 0:
        return {}
    ac = remove_dc(np.asarray(data, dtype=np.float64))
    rms_per_ch = np.sqrt(np.mean(data ** 2, axis=0))
    rms_ac_per_ch = np.sqrt(np.mean(ac**2, axis=0))
    mean_abs = np.mean(np.abs(data), axis=0)
    p5 = np.percentile(data, 5, axis=0)
    p95 = np.percentile(data, 95, axis=0)
    dynamic_range = p95 - p5

    print(f"\n=== {label} ({data.shape[0]} 样本) ===")
    print(f"{'Ch':>4} {'RMS(uV)':>10} {'MAV(uV)':>10} {'P5':>10} {'P95':>10} {'Range':>10}")
    for i in range(CHANNEL_COUNT):
        print(f"  {i+1:>2} {rms_per_ch[i]:>10.2f} {mean_abs[i]:>10.2f} "
              f"{p5[i]:>10.2f} {p95[i]:>10.2f} {dynamic_range[i]:>10.2f}")

    return {
        "rms": rms_per_ch,
        "rms_ac": rms_ac_per_ch,
        "mav": mean_abs,
        "range": dynamic_range,
        "data": data,
    }


def zc_ssc_analysis(data: np.ndarray, label: str):
    """测试不同 threshold 下的 ZC/SSC 计数。"""
    if data.shape[0] < 250:
        print(f"  数据太短，跳过 ZC/SSC 分析")
        return

    # 250 样本窗口；与网关特征一致，先按列去直流再算 ZC/SSC
    window = remove_dc(np.asarray(data[:250], dtype=np.float64))
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    print(f"\n=== ZC/SSC 阈值敏感度 ({label}, ch1, 250 样本, 已去直流) ===")
    print(f"{'Threshold':>10} {'ZC':>8} {'SSC':>8}")

    sig = window[:, 0]
    for thr in thresholds:
        # ZC
        diffs = np.diff(np.sign(sig))
        abs_diff = np.abs(np.diff(sig))
        zc = int(np.sum((diffs != 0) & (abs_diff > thr)))

        # SSC
        if len(sig) >= 3:
            d1 = sig[1:-1] - sig[:-2]
            d2 = sig[2:] - sig[1:-1]
            ssc = int(np.sum((d1 * d2 < 0) & (np.abs(d1) > thr) & (np.abs(d2) > thr)))
        else:
            ssc = 0

        marker = " <-- current" if thr == 20.0 else ""
        print(f"  {thr:>8.1f} {zc:>8} {ssc:>8}{marker}")


def adc_formula_check():
    """打印 ADC 转换公式的边界值。"""
    print("\n=== ADC 转换公式验证 ===")
    print("  WAVELETECH 说明书: 24-bit 有符号整数, 值即为 µV (固件内部已完成 ADC 换算)")
    print(f"  24-bit max (+8388607) → {_decode_signed24(0x7F, 0xFF, 0xFF):.0f} uV")
    print(f"  24-bit min (-8388608) → {_decode_signed24(0x80, 0x00, 0x00):.0f} uV")
    print(f"  raw = 1000           → {_decode_signed24(0x00, 0x03, 0xE8):.0f} uV")
    print(f"  raw = 10000          → {_decode_signed24(0x00, 0x27, 0x10):.0f} uV")
    print()
    print("  注: 旧代码使用 V_ref=4.5V/Gain=1200 换算, 已修正为直接读取 µV 值")


def main():
    parser = argparse.ArgumentParser(description="EMG 信号诊断")
    parser.add_argument("--port", required=True, help="串口设备路径")
    parser.add_argument("--duration", type=float, default=10.0, help="每阶段采集秒数")
    args = parser.parse_args()

    adc_formula_check()

    print(f"\n连接 {args.port}...")
    stream = SerialEMGStream(args.port)
    stream.start()
    time.sleep(1)

    stats = stream.frame_stats()
    print(f"初始统计: EMG={stats.emg_frames}, IMU={stats.imu_frames}, "
          f"dropped={stats.dropped_frames}")

    if stats.emg_frames == 0:
        print("警告: 1 秒内未收到 EMG 帧，请检查手环是否开启")

    # Phase 1: 放松
    input("\n按 Enter 开始「放松」阶段（手臂自然放在桌上）...")
    relax = collect_phase(stream, args.duration, "放松 (baseline)")
    relax_stats = analyze(relax, "放松")

    # Phase 2: 握拳
    input("\n按 Enter 开始「握拳」阶段（持续用力握拳）...")
    grip = collect_phase(stream, args.duration, "握拳 (contraction)")
    grip_stats = analyze(grip, "握拳")

    # Phase 3: 打字
    input("\n按 Enter 开始「打字」阶段（正常打字 10 秒）...")
    typing = collect_phase(stream, args.duration, "打字")
    typing_stats = analyze(typing, "打字")

    # 对比
    if relax_stats and grip_stats:
        print("\n=== 信号质量诊断 ===")
        ratio_raw = grip_stats["rms"] / (relax_stats["rms"] + 1e-12)
        ratio_ac = grip_stats["rms_ac"] / (relax_stats["rms_ac"] + 1e-12)
        ratio_range = grip_stats["range"] / (relax_stats["range"] + 1e-12)
        print("  握拳/放松 比值 (强 DC 时请看 AC-RMS 与 Range，勿单看原始 RMS):")
        print(f"  {'Ch':>4} {'RMS(raw)':>10} {'RMS(AC)':>10} {'Range':>10}")
        for i in range(CHANNEL_COUNT):
            ok = ratio_ac[i] > 3 or ratio_range[i] > 3
            weak = ratio_ac[i] > 1.5 or ratio_range[i] > 1.5
            quality = "OK" if ok else ("弱" if weak else "无信号")
            print(
                f"    {i+1:>2} {ratio_raw[i]:>10.1f} {ratio_ac[i]:>10.1f} "
                f"{ratio_range[i]:>10.1f}  [{quality}]"
            )

        mean_ac = float(np.mean(ratio_ac))
        if mean_ac < 1.5:
            print(f"\n  去直流后 RMS 比值均值 {mean_ac:.1f}x — 可动幅度仍极弱")
            print("  可能原因:")
            print("    1. 电极接触不良 / 手环太松 / 未做目标动作")
            print("    2. 串口被其他进程占用，采集样本过少")
        elif mean_ac < 3:
            print(f"\n  去直流后 RMS 比值均值 {mean_ac:.1f}x — 偏弱但可用")
        else:
            print(f"\n  去直流后 RMS 比值均值 {mean_ac:.1f}x — 良好")

    # ZC/SSC 阈值分析
    if relax_stats:
        zc_ssc_analysis(relax_stats["data"], "放松")
    if grip_stats:
        zc_ssc_analysis(grip_stats["data"], "握拳")

    # 最终统计
    stats = stream.frame_stats()
    print(f"\n=== 最终流统计 ===")
    print(f"  EMG: {stats.emg_frames}, IMU: {stats.imu_frames}, "
          f"Dropped(EMG only): {stats.dropped_frames}")

    stream.stop()
    print("\n诊断完成。")


if __name__ == "__main__":
    main()
