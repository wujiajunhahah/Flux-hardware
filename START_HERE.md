# Desktop Gesture Trainer -- Quick Start

> This is the **desktop serial gesture application** (root `app.py`).
> If you want the **Web API + iOS phone connection**, see the main [README.md](README.md) and run `python web/app.py` instead.

```bash
cd /path/to/harward-gesture          # enter repo root
source .venv/bin/activate             # activate venv
python3 app.py --port /dev/cu.usbserial-0001 --baud 921600 --fs 1000
```

If your serial port name is different:

```bash
ls /dev/cu.usbserial*
```
