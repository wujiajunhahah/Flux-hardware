#!/usr/bin/env python3
"""
FluxChi -- Flywheel 训练管线 (Feature -> XGBoost -> ONNX)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flywheel.store import FlywheelStore


def build_matrix(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_keys = sorted({k for s in samples for k in s.get("features", {}).keys()})
    x = np.zeros((len(samples), len(feature_keys)), dtype=np.float32)
    y = np.array([s["label"] for s in samples], dtype=object)
    key_to_i = {k: i for i, k in enumerate(feature_keys)}
    for row_i, s in enumerate(samples):
        for k, v in s.get("features", {}).items():
            if k in key_to_i:
                x[row_i, key_to_i[k]] = float(v)
    return x, y, feature_keys


def export_onnx(model, n_features: int, out_path: Path) -> bool:
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("features", FloatTensorType([None, n_features]))]
        onx = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        out_path.write_bytes(onx.SerializeToString())
        return True
    except Exception as exc:
        print(f"[WARN] ONNX 导出失败: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flywheel model with XGBoost")
    parser.add_argument("--db", default="~/.fluxchi/flywheel.db", help="Flywheel SQLite path")
    parser.add_argument("--model-dir", default="model", help="输出目录")
    parser.add_argument("--prefix", default="flywheel_xgb", help="模型名前缀")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = FlywheelStore(args.db)
    samples = store.export_training_data()
    store.close()
    if len(samples) < 40:
        raise SystemExit(f"[ERROR] 样本过少: {len(samples)}，至少需要 40 条带标签快照")

    x, y, feature_keys = build_matrix(samples)
    labels = sorted(set(y.tolist()))
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    y_id = np.array([label_to_id[v] for v in y], dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_id, test_size=args.test_size, random_state=args.seed, stratify=y_id
    )

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise SystemExit(f"[ERROR] 缺少 xgboost，请先 pip install xgboost: {exc}") from exc

    n_classes = len(labels)
    xgb_kwargs = dict(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=args.seed,
    )
    if n_classes <= 1:
        raise SystemExit("[ERROR] 训练数据仅包含 1 个类别，无法训练分类器")
    if n_classes == 2:
        xgb_kwargs.update(objective="binary:logistic", eval_metric="logloss")
    else:
        xgb_kwargs.update(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)

    model = XGBClassifier(**xgb_kwargs)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="weighted"))
    report = classification_report(y_test, pred, output_dict=True)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_json = out / f"{args.prefix}.json"
    model.save_model(model_json)
    onnx_path = out / f"{args.prefix}.onnx"
    onnx_ok = export_onnx(model, x.shape[1], onnx_path)

    config = {
        "model_type": "XGBoostClassifier",
        "classes": labels,
        "class_to_id": label_to_id,
        "n_features": int(x.shape[1]),
        "feature_names": feature_keys,
        "metrics": {
            "accuracy": acc,
            "weighted_f1": f1,
            "report": report,
        },
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "onnx_exported": bool(onnx_ok),
    }
    (out / f"{args.prefix}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[done] model={model_json}")
    print(f"[done] onnx={onnx_path if onnx_ok else 'FAILED'}")
    print(f"[done] config={out / f'{args.prefix}_config.json'}")
    print(f"[metrics] acc={acc:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()

