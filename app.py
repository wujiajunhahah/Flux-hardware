#!/usr/bin/env python3
"""Waveletech EMG gesture training and recognition app."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Optional

import matplotlib.pyplot as plt

from src.actions import GestureActions
from src.bridge import stream_to_libemg
from src.features import FeatureExtractor, estimate_contact_quality
from src.inference import GestureInference, PredictionResult
from src.recorder import GestureRecorder
from src.ring_buffer import EMGRingBuffer
from src.state import StateEstimator, StateReadings
from src.stream import BaseEMGStream, SerialEMGStream
from src.trainer import Trainer, load_model
from src.ui import EMGPlotUI


class EMGGestureApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sample_rate = args.fs
        self.stream = self._create_stream()
        self.buffer = EMGRingBuffer(max_seconds=args.history, sample_rate=self.sample_rate)
        self.recorder = GestureRecorder(base_dir=args.data_dir)
        self.trainer = Trainer(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            sample_rate=self.sample_rate,
            window_seconds=args.window,
            stride_seconds=args.stride,
        )
        self.actions = GestureActions(args.gestures)
        self.inference: Optional[GestureInference] = None
        self.inference_enabled = False
        self.latest_prediction: Optional[PredictionResult] = None
        self.state_estimator = StateEstimator(
            sample_rate=self.sample_rate,
            window_seconds=args.state_window,
            fatigue_history=args.fatigue_history,
            imu_history=args.imu_history,
        )
        self.latest_state: Optional[StateReadings] = None
        self.libemg_bridge = stream_to_libemg()
        self.ui = EMGPlotUI(window_seconds=args.history)
        self.ui.connect_keypress(self._on_key)
        self.ui.set_actions(
            record=self._toggle_recording,
            train=self._start_training_thread,
            inference=self._toggle_inference,
            screenshot=self._save_screenshot,
            quit=lambda: plt.close(self.ui.fig),
        )

        self._samples_counter = 0
        self._samples_per_second = 0.0
        self._rate_start = time.time()
        self._load_model()

    def _create_stream(self) -> BaseEMGStream:
        port = self.args.port
        if not port:
            raise SystemExit("[app] Serial port is required (use --port).")
        print(f"[app] Connecting to {port} @ {self.args.baud} baud...")
        return SerialEMGStream(port, self.args.baud)

    def run(self) -> None:
        try:
            self.stream.start()
        except RuntimeError as exc:
            raise SystemExit(f"[app] Failed to start stream: {exc}") from exc
        print("[app] Shortcuts: R=record, T=train, I=inference, S=screenshot, Q=quit")
        self.ui.start_animation(self._update_plot)
        try:
            self.ui.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if hasattr(self, "libemg_bridge") and self.libemg_bridge:
            self.libemg_bridge.close()
        self.stream.stop()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "q":
            plt.close(self.ui.fig)
        elif key == "r":
            self._toggle_recording(label=self.ui.get_label_text())
        elif key == "t":
            self._start_training_thread()
        elif key == "i":
            self._toggle_inference()
        elif key == "s":
            self._save_screenshot()

    def _toggle_recording(self, label: Optional[str] = None) -> None:
        if self.recorder.is_recording():
            self.recorder.stop()
            self.ui.set_record_button_state(False, "")
            return
        actual_label = label.strip() if label else ""
        if not actual_label:
            actual_label = "gesture"
        try:
            self.recorder.start(actual_label)
            self.ui.set_record_button_state(True, actual_label)
        except ValueError as exc:
            print(f"[recorder] {exc}")

    def _start_training_thread(self) -> None:
        threading.Thread(target=self._train_model, daemon=True).start()

    def _save_screenshot(self) -> None:
        path = self.ui.save_screenshot()
        print(f"[app] Screenshot saved to {path}")

    def _toggle_inference(self) -> None:
        if not self.inference:
            print("[inference] No trained model found. Press T after recording gestures.")
            return
        self.inference_enabled = not self.inference_enabled
        state = "ON" if self.inference_enabled else "OFF"
        print(f"[inference] Real-time inference {state}")

    def _train_model(self) -> None:
        print("[trainer] Training model...")
        metrics = self.trainer.train()
        if metrics:
            self._load_model(reload_only=True)

    # ------------------------------------------------------------------
    # Model / inference management
    # ------------------------------------------------------------------
    def _load_model(self, reload_only: bool = False) -> None:
        model, config = load_model(self.args.model_dir)
        if not model or not config:
            if not reload_only:
                print("[inference] No trained model found yet.")
            self.inference = None
            self.inference_enabled = False
            return

        extractor = FeatureExtractor(
            sample_rate=int(config.get("sample_rate", self.sample_rate)),
            window_seconds=float(config.get("window_seconds", self.args.window)),
            stride_seconds=float(config.get("stride_seconds", self.args.stride)),
        )
        self.inference = GestureInference(
            model=model,
            extractor=extractor,
            stable_windows=self.args.stable_windows,
            threshold=self.args.threshold,
        )
        print("[inference] Model loaded. Press I to enable real-time inference.")

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------
    def _update_plot(self, frame):  # Matplotlib animation signature
        samples = self.stream.consume_samples()
        if samples:
            self.libemg_bridge.submit(samples)
            self.buffer.extend(samples)
            self.recorder.append(samples)
            self._update_sample_rate(len(samples))
            if self.inference_enabled and self.inference:
                results = self.inference.process(samples)
                for result in results:
                    self.latest_prediction = result
                    if result.stable:
                        self.actions.execute(result.label, stream=self.stream)

        imu_samples = self.stream.consume_imu()
        if imu_samples:
            self.state_estimator.update_imu(imu_samples)

        times, data = self.buffer.to_arrays()
        normalized = self.buffer.normalize(data)
        self.ui.update_waveforms(times, normalized)

        state = self.state_estimator.update(times, data)
        if state is not None:
            self.latest_state = state

        rms_values = self.buffer.rms(self.args.rms_window)
        qualities = [estimate_contact_quality(val) for val in rms_values]
        self.ui.update_rms(rms_values, qualities)

        status = self._status_text()
        recording = f"Recording: {self.recorder.label or 'OFF'}"
        inference = f"Inference: {'ON' if self.inference_enabled and self.inference else 'OFF'}"
        prediction = self._prediction_text()
        states = self._state_text()
        self.ui.update_status(status, recording, inference, prediction, states)

    def _prediction_text(self) -> str:
        if not self.latest_prediction:
            return "Prediction: --"
        label = self.latest_prediction.label
        confidence = self.latest_prediction.confidence
        stable = " *" if self.latest_prediction.stable else ""
        return f"Prediction: {label} ({confidence:.2f}){stable}"

    def _status_text(self) -> str:
        stats = self.stream.frame_stats()
        return (
            f"Sample rate: {self.sample_rate} Hz | samples/s: {self._samples_per_second:.0f}"
            f" | frames {stats.emg_frames} EMG / {stats.imu_frames} IMU | dropped: {stats.dropped_frames}"
        )

    def _state_text(self) -> str:
        if not self.latest_state:
            return "States: fatigue -- | anxiety -- | focus --"
        state = self.latest_state
        return (
            f"States: fatigue {state.fatigue:.2f} | anxiety {state.anxiety:.2f}"
            f" | focus {state.focus:.2f}"
        )

    def _update_sample_rate(self, new_samples: int) -> None:
        self._samples_counter += new_samples
        now = time.time()
        elapsed = now - self._rate_start
        if elapsed >= 1.0:
            self._samples_per_second = self._samples_counter / elapsed
            self._samples_counter = 0
            self._rate_start = now


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Waveletech EMG gesture tool")
    parser.add_argument("--port", default="/dev/cu.usbserial-0001", help="Serial port (e.g., /dev/cu.usbserial-0001)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate for serial streaming")
    parser.add_argument("--fs", type=int, default=1000, help="Sampling rate (Hz)")
    parser.add_argument("--window", type=float, default=0.5, help="Feature window length (seconds)")
    parser.add_argument("--stride", type=float, default=0.1, help="Feature stride (seconds)")
    parser.add_argument("--history", type=float, default=8.0, help="Seconds of waveform history to display")
    parser.add_argument("--rms-window", type=float, default=0.2, dest="rms_window", help="Seconds for RMS computation")
    parser.add_argument("--state-window", type=float, default=0.5, help="Window length for psychophysiological estimates")
    parser.add_argument("--fatigue-history", type=float, default=60.0, dest="fatigue_history", help="History horizon (s) for fatigue slope")
    parser.add_argument("--imu-history", type=float, default=5.0, dest="imu_history", help="Seconds of IMU data for jerk computation")
    parser.add_argument("--data-dir", default="data", help="Directory for gesture recordings")
    parser.add_argument("--model-dir", default="model", help="Directory for trained model files")
    parser.add_argument("--gestures", default="gestures.yaml", help="Gesture-to-action mapping file")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold for triggers")
    parser.add_argument("--stable-windows", type=int, default=3, dest="stable_windows", help="Consecutive windows required for trigger")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    app = EMGGestureApp(args)
    app.run()


if __name__ == "__main__":
    main(sys.argv[1:])
