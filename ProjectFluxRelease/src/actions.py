"""Gesture action execution loaded from gestures.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import urllib.request
import urllib.error

import yaml

from .stream import BaseEMGStream


def load_actions(path: str = "gestures.yaml") -> Dict[str, Dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        with config_path.open() as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - config errors at runtime
        print(f"[actions] Failed to load {config_path}: {exc}")
        return {}
    actions: Dict[str, Dict[str, Any]] = {}
    for gesture, spec in data.items():
        if not isinstance(spec, dict):
            continue
        actions[str(gesture)] = spec
    return actions


class GestureActions:
    """Executes configured actions for detected gestures."""

    def __init__(self, path: str = "gestures.yaml") -> None:
        self.path = Path(path)
        self._actions = load_actions(path)

    def reload(self) -> None:
        self._actions = load_actions(str(self.path))

    def action_for(self, gesture: str) -> Optional[Dict[str, Any]]:
        return self._actions.get(gesture)

    def execute(self, gesture: str, stream: Optional[BaseEMGStream] = None) -> None:
        spec = self.action_for(gesture)
        if not spec:
            print(f"TRIGGER:{gesture}")
            return

        action = spec.get("action", "print").lower()
        if action == "print":
            message = spec.get("message") or f"TRIGGER:{gesture}"
            print(message)
        elif action == "serial":
            command = spec.get("command", "")
            if not command:
                print(f"[actions] serial action for {gesture} missing 'command'")
                return
            if stream is not None:
                stream.send_command(command)
            print(f"TRIGGER:{gesture} -> serial:{command}")
        elif action == "http":
            url = spec.get("url")
            if not url:
                print(f"[actions] http action for {gesture} missing 'url'")
                return
            payload = spec.get("payload")
            data: Optional[bytes] = None
            headers = {"Content-Type": "application/json"}
            if payload is not None:
                if isinstance(payload, (dict, list)):
                    data = json.dumps(payload).encode("utf-8")
                else:
                    data = str(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=2):
                    pass
                print(f"TRIGGER:{gesture} -> http:{url}")
            except urllib.error.URLError as exc:
                print(f"[actions] HTTP request failed for {gesture}: {exc}")
        else:
            print(f"[actions] Unknown action '{action}' for {gesture}")
