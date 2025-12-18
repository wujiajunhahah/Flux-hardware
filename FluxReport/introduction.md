# Project Flux Overview

Project Flux is a Mac-friendly EMG and IMU processing stack designed for the Waveletech / OYMotion 8-channel wristband. The system captures AA/BB frames, decodes 24-bit signed EMG samples, normalizes them into microvolts, visualizes waveforms + RMS bars, and computes psychophysiological states (fatigue, anxiety, focus).

## Hardware Notes
- Serial port: `/dev/cu.usbserial-0001`, 921600 baud, 29-byte frames.
- `0xAA` frames contain eight 24-bit EMG channels.
- `0xBB` frames contain 6-axis IMU data; gyro immediately follows accel.
- EMG values are scaled by `(raw / 2^23-1) * (4.5 V / 1200 gain) * 1e6` to produce microvolts.

## Final Feature Set
- Time domain HTD features (MAV/RMS/WL/ZC/SSC) with spatial correlations for micro-gesture recognition.
- LibEMG bridge via UDP to 127.0.0.1:12345 for live interoperability.

## Psychophysiological States
- **Fatigue**: Mean Power Frequency (MNF) slope over 60s.
- **Anxiety**: Co-contraction index + IMU jerk.
- **Focus**: Autocorrelation-based rhythmicity scaled by (1 - fatigue)*(1 - anxiety).
