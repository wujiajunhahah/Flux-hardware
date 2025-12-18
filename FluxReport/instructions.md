# Usage Instructions

1. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Verify device streaming:
   ```bash
   python3 -m tools.serial_probe --port /dev/cu.usbserial-0001 --seconds 5
   ```
3. Launch Project Flux:
   ```bash
   python3 app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
   ```

## Recording
- Use the Label text box to set gesture name.
- Click **Record** to start; click again to stop (button text toggles to `Stop (label)`).
- Repeat for every custom gesture.

## Training & Inference
- Click **Train** (or press `T`) to build a RandomForest model.
- After training, click **Inference** (`I`) to enable live predictions.
- Status panel shows fatigue/anxiety/focus, prediction box shows label + confidence.

## LibEMG Bridge
- EMG samples stream via UDP (`127.0.0.1:12345`) so LibEMG tools can attach.
- All HTD feature extraction (plus fallback) is encapsulated in `src/features.py`.
