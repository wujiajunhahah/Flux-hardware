"""Session review helpers for archived FluxChi recordings."""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_session_review(pkg: Dict[str, Any]) -> Dict[str, Any]:
    snaps: List[Dict[str, Any]] = list(pkg.get("snapshots") or [])
    if not snaps:
        return {
            "summary": {
                "average_stamina": 0.0,
                "largest_drop": 0,
                "recovered": False,
                "duration_min": 0.0,
                "net_change": 0,
            },
            "moments": {},
            "flags": [],
            "series": [],
        }

    staminas = [_safe_float(s.get("stamina")) for s in snaps]
    tensions = [_safe_float(s.get("ten")) for s in snaps]
    fatigues = [_safe_float(s.get("fat")) for s in snaps]
    times = [_safe_float(s.get("t")) for s in snaps]

    highest_idx = max(range(len(staminas)), key=staminas.__getitem__)
    lowest_idx = min(range(len(staminas)), key=staminas.__getitem__)
    largest_drop = round(staminas[highest_idx] - staminas[lowest_idx])
    recovered = lowest_idx < len(staminas) - 1 and (max(staminas[lowest_idx + 1 :]) - staminas[lowest_idx]) >= 10
    duration_min = round((_safe_float((pkg.get("session") or {}).get("durationSec")) or times[-1]) / 60.0, 1)

    flags = []
    high_tension_count = sum(1 for value in tensions if value >= 0.55)
    if high_tension_count:
        flags.append(
            {
                "type": "high_tension",
                "minutes": max(high_tension_count - 1, 1),
                "message": "张力段偏高，建议检查姿势、击键力度或休息节奏。",
            }
        )

    steepest_delta = 0.0
    steepest_idx = 0
    for idx in range(1, len(staminas)):
        delta = staminas[idx - 1] - staminas[idx]
        if delta > steepest_delta:
            steepest_delta = delta
            steepest_idx = idx

    review = {
        "summary": {
            "average_stamina": round(sum(staminas) / len(staminas), 1),
            "average_tension": round(sum(tensions) / len(tensions), 3),
            "average_fatigue": round(sum(fatigues) / len(fatigues), 3),
            "largest_drop": largest_drop,
            "recovered": recovered,
            "duration_min": duration_min,
            "net_change": round(staminas[-1] - staminas[0]),
        },
        "moments": {
            "lowest": {
                "index": lowest_idx,
                "minute": round(times[lowest_idx] / 60.0, 1),
                "stamina": round(staminas[lowest_idx]),
            },
            "highest": {
                "index": highest_idx,
                "minute": round(times[highest_idx] / 60.0, 1),
                "stamina": round(staminas[highest_idx]),
            },
            "steepest_drop": {
                "index": steepest_idx,
                "minute": round(times[steepest_idx] / 60.0, 1),
                "delta": round(steepest_delta, 1),
            },
        },
        "flags": flags,
        "series": [
            {
                "minute": round(times[idx] / 60.0, 1),
                "stamina": staminas[idx],
                "tension": tensions[idx],
                "fatigue": fatigues[idx],
                "state": snaps[idx].get("state", "unknown"),
                "activity": snaps[idx].get("act", "unknown"),
            }
            for idx in range(len(snaps))
        ],
    }
    return review
