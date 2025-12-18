# Waveletech Gesture Lab

Complete macOS-friendly toolkit for recording, training, and recognizing custom EMG gestures with the Waveletech 8-channel wristband. The app ingests the binary AA/BB frames (0xD2 headers) via serial, plots real-time waveforms and RMS bars, records labeled gestures, trains a RandomForest model, and runs low-latency inference with trigger actions.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt
```

## Quick start

1. Connect the Waveletech USB receiver (shows up as `/dev/cu.usbserial-0001`).
2. Launch the app (fallbacks to synthetic data if no port is provided):

   ```bash
   python3 app.py --port /dev/cu.usbserial-0001 --baud 921600
   ```

3. Use the keyboard shortcuts while the Matplotlib window is focused:

   | Key | Action |
   | --- | ------ |
   | `R` | Start/stop recording the current gesture (prompts for a label in the terminal). |
   | `T` | Train a RandomForest model from all recordings in `data/`. |
   | `I` | Toggle real-time inference (requires a trained model). |
   | `S` | Save a screenshot to `screenshots/`. |
   | `Q` | Quit the app. |

4. During inference, stable detections (≥3 consecutive windows with confidence ≥0.8) print `TRIGGER:<gesture>` and run the configured action from `gestures.yaml`.

### Workflow

1. **Record**: Press `R`, enter a gesture label, perform the gesture, then press `R` again. Files are saved to `data/<gesture>/<timestamp>.csv` with columns `t, ch1..ch8`.
2. **Train**: Press `T`. The trainer extracts MAV/RMS/WL/ZC/SSC features per channel using 500 ms windows with 100 ms stride, splits data by file (GroupShuffleSplit when possible), and writes `model/model.pkl` + `model/config.json`.
3. **Infer**: Press `I` to enable inference. The UI shows the latest prediction + confidence, along with RMS-based contact quality (`GOOD/WEAK/NOISY`) for each channel. When a trigger fires, optional serial/HTTP actions run (see below).

### Command-line options

`python3 app.py --help` lists all tunables, including display history (`--history`), RMS window (`--rms-window`), feature window (`--window`), stride (`--stride`), alternate data/model directories, and `--demo` for synthetic playback.

## Gesture actions (`gestures.yaml`)

Map gesture names to actions:

```yaml
gesture_name:
  action: print            # print | serial | http
  message: "TRIGGER:gesture"

fist:
  action: serial
  command: "FIST\n"       # sent back over the same serial port

wave:
  action: http
  url: "http://localhost:7000/gesture"
  payload:
    event: wave
```

- `print`: logs the message.
- `serial`: sends `command` bytes to the wristband receiver (also prints a summary).
- `http`: issues a JSON POST to `url` with the optional `payload`.

Update the file and press `I` again (or restart) to reload.

## Sanity probe

`tools/serial_probe.py` validates connectivity at 921 600 baud before launching the UI:

```bash
python3 tools/serial_probe.py --port /dev/cu.usbserial-0001 --seconds 10
```

Output includes AA/BB frame ratios, dropped sequence counts, and EMG frame throughput.

## Repository layout

```
app.py                  # Main entrypoint / keyboard controls / Matplotlib loop
src/stream.py           # Serial + synthetic readers, packet parsing, frame stats
src/ring_buffer.py      # Multichannel ring buffer and RMS helper
src/features.py         # Gesture feature extraction + contact quality
src/trainer.py          # Recording loader + RandomForest trainer + persistence
src/recorder.py         # CSV recordings per gesture
src/inference.py        # Real-time windowing, smoothing, and trigger logic
src/actions.py          # gestures.yaml loader + serial/HTTP execution
src/ui.py               # Matplotlib figure, waveform + RMS bars, status text
tools/serial_probe.py   # Standalone serial link checker
gestures.yaml           # Default gesture→action map
```

Legacy prototypes (`visualizer.py`, `emg_rms_realtime.py`, etc.) remain for reference but the recommended entrypoint is always `python3 app.py ...`.
