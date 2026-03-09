#!/usr/bin/env python3
"""
Train baseline FluxStamina CoreML updatable model.

Creates a small MLP for on-device personalization via MLUpdateTask.

Usage:
    pip install coremltools numpy
    python scripts/train_stamina_model.py

Output:
    ios/FluxChi/ML/FluxStamina.mlmodel
"""

import os
import numpy as np

try:
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
except ImportError:
    print("Install coremltools first: pip install coremltools")
    raise SystemExit(1)


def create_updatable_model():
    input_dim = 6

    input_features = [("features", ct.models.datatypes.Array(input_dim))]
    output_features = [("stamina", ct.models.datatypes.Double())]

    builder = NeuralNetworkBuilder(
        input_features, output_features,
        disable_rank5_shape_mapping=True
    )

    np.random.seed(42)

    # Dense 1: 6 -> 16
    builder.add_inner_product(
        name="dense1",
        W=np.random.randn(16, input_dim).astype(np.float32) * 0.1,
        b=np.zeros(16, dtype=np.float32),
        input_channels=input_dim, output_channels=16,
        has_bias=True,
        input_name="features", output_name="dense1_out"
    )
    builder.add_activation(
        name="relu1", non_linearity="RELU",
        input_name="dense1_out", output_name="relu1_out"
    )

    # Dense 2: 16 -> 8
    builder.add_inner_product(
        name="dense2",
        W=np.random.randn(8, 16).astype(np.float32) * 0.1,
        b=np.zeros(8, dtype=np.float32),
        input_channels=16, output_channels=8,
        has_bias=True,
        input_name="relu1_out", output_name="dense2_out"
    )
    builder.add_activation(
        name="relu2", non_linearity="RELU",
        input_name="dense2_out", output_name="relu2_out"
    )

    # Dense 3 (output): 8 -> 1
    builder.add_inner_product(
        name="dense3",
        W=np.random.randn(1, 8).astype(np.float32) * 0.1,
        b=np.array([0.5], dtype=np.float32),
        input_channels=8, output_channels=1,
        has_bias=True,
        input_name="relu2_out", output_name="raw_out"
    )
    builder.add_activation(
        name="sigmoid", non_linearity="SIGMOID",
        input_name="raw_out", output_name="stamina"
    )

    # Mark layers as updatable
    builder.make_updatable(["dense1", "dense2", "dense3"])

    spec = builder.spec

    # Training input
    ti = spec.description.trainingInput.add()
    ti.name = "stamina_target"
    ti.type.doubleType.SetInParent()

    # MSE loss
    loss = spec.updatable.loss.add()
    loss.name = "mse"
    loss.meanSquaredError.input = "stamina"
    loss.meanSquaredError.target = "stamina_target"

    # SGD optimizer
    sgd = spec.updatable.optimizer.sgdOptimizer
    sgd.learningRate.defaultValue = 0.01
    sgd.learningRate.range.minValue = 0.001
    sgd.learningRate.range.maxValue = 0.1
    sgd.miniBatchSize.defaultValue = 4
    sgd.momentum.defaultValue = 0.0

    # Epochs
    spec.updatable.epochs.defaultValue = 10
    spec.updatable.epochs.set.values.extend([5, 10, 20])

    return ct.models.MLModel(spec)


if __name__ == "__main__":
    print("Creating FluxStamina updatable CoreML model...")
    model = create_updatable_model()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "ios", "FluxChi", "ML")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "FluxStamina.mlmodel")
    model.save(out_path)
    print(f"Saved to {out_path}")
