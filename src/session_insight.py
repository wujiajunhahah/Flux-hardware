"""
会话级「洞察」——确定性规则，无 UI 逻辑；供多端共用同一后端输出。
"""

from __future__ import annotations

from typing import Any, Dict, List


def session_insight_text(pkg: Dict[str, Any]) -> str:
    """从归档包生成简短中文结论（可后续换模型，接口保持不变）。"""
    snaps: List[Dict[str, Any]] = pkg.get("snapshots") or []
    if not snaps:
        return "暂无快照，无法生成洞察。"

    stamins = [float(s.get("stamina", 0) or 0) for s in snaps if isinstance(s, dict)]
    if not stamins:
        return "快照中缺少耐力字段。"

    avg = sum(stamins) / len(stamins)
    mn, mx = min(stamins), max(stamins)
    dur_sec = float(snaps[-1].get("t", 0) or 0)
    parts = [
        f"时长约 {dur_sec / 60:.1f} 分钟，采样 {len(snaps)} 点。",
        f"耐力平均 {avg:.0f}（区间 {mn:.0f}–{mx:.0f}）。",
    ]
    if mx - mn > 35:
        parts.append("波动偏大，建议观察休息是否规律。")
    elif mx - mn < 8 and dur_sec > 300:
        parts.append("曲线较平，可能处于稳定专注或传感器变化较小。")

    tens = [float(s.get("ten", 0) or 0) for s in snaps if isinstance(s, dict)]
    if tens:
        tavg = sum(tens) / len(tens)
        if tavg > 0.45:
            parts.append("张力指标偏高，注意肩颈与用力方式。")

    return " ".join(parts)
