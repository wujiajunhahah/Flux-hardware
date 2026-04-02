"""
会话级「洞察」——确定性规则，无 UI 逻辑；供多端共用同一后端输出。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .session_review import build_session_review


def session_insight_text(pkg: Dict[str, Any]) -> str:
    """从归档包生成简短中文结论（可后续换模型，接口保持不变）。"""
    snaps: List[Dict[str, Any]] = pkg.get("snapshots") or []
    if not snaps:
        return "暂无快照，无法生成洞察。"

    review = build_session_review(pkg)
    summary = review.get("summary") or {}
    if not review.get("series"):
        return "快照中缺少耐力字段。"

    parts = [
        f"时长约 {float(summary.get('duration_min', 0.0)):.1f} 分钟，采样 {len(snaps)} 点。",
        f"耐力平均 {float(summary.get('average_stamina', 0.0)):.0f}。",
        f"最大下滑 {float(summary.get('largest_drop', 0.0)):.0f}。",
    ]
    if float(summary.get("largest_drop", 0.0)) > 35:
        parts.append("波动偏大，建议观察休息是否规律。")
    elif float(summary.get("largest_drop", 0.0)) < 8 and float(summary.get("duration_min", 0.0)) > 5:
        parts.append("曲线较平，可能处于稳定专注或传感器变化较小。")

    if summary.get("recovered"):
        parts.append("后段出现恢复。")
    else:
        parts.append("后段恢复不明显。")

    if float(summary.get("average_tension", 0.0)) > 0.45:
        parts.append("张力指标偏高，注意肩颈与用力方式。")

    return " ".join(parts)
