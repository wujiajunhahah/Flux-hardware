"""
多端 EMG 对齐：统一为 8 路 canonical RMS（与 USB 串口帧一致）。

- 手机 BLE：通常为 6 路有效，第 7–8 路在后端补 0，并在 signalMeta 中声明来源。
- 分析脚本只需读固定宽度 8，由 meta 判断哪些是「真实通道」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

CANONICAL_EMG_CHANNELS = 8


def normalize_rms_to_canonical(
    rms: Optional[Sequence[Union[int, float]]],
    *,
    source_channels: Optional[int] = None,
) -> List[float]:
    """将任意长度的 RMS 向量规范为长度 8；不足补 0，超出截断。"""
    if rms is None:
        raw: List[float] = []
    else:
        raw = [float(x) for x in rms]
    out = raw[:CANONICAL_EMG_CHANNELS]
    while len(out) < CANONICAL_EMG_CHANNELS:
        out.append(0.0)
    _ = source_channels  # 预留：未来可做电极重映射
    return out


def canonicalize_export_package(pkg: Dict[str, Any]) -> Dict[str, Any]:
    """
    就地规范化 iOS ExportPackage 或同构 JSON：每条 snapshot 的 rms 为 8 维；
    更新 signalMeta.canonical* 字段。
    """
    meta = pkg.get("signalMeta")
    if not isinstance(meta, dict):
        meta = {}
        pkg["signalMeta"] = meta

    snaps = pkg.get("snapshots")
    if not isinstance(snaps, list):
        return pkg

    active = int(meta.get("activeChannels") or meta.get("active_channels") or 0)
    if active <= 0 and snaps:
        first = snaps[0]
        if isinstance(first, dict) and isinstance(first.get("rms"), list):
            active = len(first["rms"])

    for s in snaps:
        if not isinstance(s, dict):
            continue
        r = s.get("rms")
        if not isinstance(r, list):
            r = []
        s["rms"] = normalize_rms_to_canonical(r, source_channels=active or len(r))

    meta["canonicalEmgChannels"] = CANONICAL_EMG_CHANNELS
    meta["padPolicy"] = "zeros_trailing_for_missing; BLE_6ch_maps_to_first_6"
    if active and active < CANONICAL_EMG_CHANNELS:
        meta["note"] = (
            f"{meta.get('note', '')} "
            f"[canonicalized] 源有效通道约 {active}，第 {active + 1}–{CANONICAL_EMG_CHANNELS} 路已补 0。"
        ).strip()

    return pkg
