# AI Antenna Tracker

Real-time object detection and auto-tracking for a pan/tilt antenna mount. Uses a webcam + YOLO to detect objects, then drives RC servo motors via MAVLink `DO_SET_SERVO` commands to keep the antenna aimed at the target.

## How it works

```
Camera → YOLO detection → Click target → Feature extraction → Frame-by-frame matching → Pan/tilt calculation → MAVLink servo commands
```

1. **Camera** captures frames via DirectShow (auto-detects physical cameras, skips virtual/NDI devices).
2. **YOLO** (Ultralytics YOLO11n) detects objects on each frame with configurable confidence.
3. **Click** an object to select it — the tracker locks on and extracts multi-modal features (color histograms, Hu moments, contour shape, edge density, gradient stats).
4. **Matching** tracks the object frame-to-frame using IOU + feature scoring. If lost, a re-acquisition stage searches same-class detections within a predicted motion window (90-frame memory).
5. **Pan/tilt** is computed proportionally to the object's offset from frame center, clamped to ±45°.
6. **Servo commands** are sent over MAVLink (UDP or serial) to move the antenna mount.

No hardware? Runs in mock mode — prints servo commands to console instead.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Click any detected object to start tracking. Press `q` to quit, `c` to clear selection.

## Usage

```bash
python main.py --camera-index 0              # specific camera
python main.py --model models/yolo11n.pt     # custom model
python main.py --conf-threshold 0.5          # higher confidence cutoff
python main.py --port COM3 --baud 57600      # serial tracker connection
python main.py --pan-channel 1 --tilt-channel 2  # servo channel mapping
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--camera-index` | auto | Camera device index (auto-detects if omitted) |
| `--conf-threshold` | 0.3 | YOLO confidence threshold |
| `--model` | models/yolo11n.pt | Path to YOLO model |
| `--port` | COM3 | Tracker serial port (ignored if pymavlink not installed) |
| `--baud` | 57600 | Serial baud rate |
| `--pan-channel` | 1 | Servo channel for pan |
| `--tilt-channel` | 2 | Servo channel for tilt |

Tracker auto-falls back to mock mode if `pymavlink` is missing or the connection fails.

## Interface

Two windows open:

- **Object Detection** — camera feed with YOLO bounding boxes. Green crosshair at center, green line to selected object.
- **Tracker Scope** — radar-style scope showing object position relative to center, with N/E/S/W cardinal directions, range rings, and a pulsing target dot. Below: pan/tilt gauge bars with numerical readout, a servo position indicator, direction arrows (▲ ▼ ◄ ►), and a status bar.

Both windows show: class name, tracking confidence %, FPS, pan/tilt angle, and TRACKING/LOST/CENTERED status.

## Project structure

```
├── main.py                    Entry point
├── config.py                  All tunable parameters
├── requirements.txt
├── vision/
│   ├── camera.py              DirectShow camera auto-detect + capture
│   ├── detector.py            YOLO detection, click-to-select, tracking, re-acquisition
│   └── matching.py            Feature extraction and scoring (color, shape, contour, etc.)
├── hardware/
│   └── tracker.py             MAVLink DO_SET_SERVO + mock fallback
├── ui/
│   └── visualizer.py          Tracker Scope window (radar, gauges, direction arrows)
├── models/
│   └── yolo11n.pt             YOLO11 nano model
└── tools/                     Test/utility scripts
```

## Configuration

All tuning is in `config.py`:

- **Detection** — confidence, IOU, minimum box area, inference size
- **Tracking** — match thresholds, memory length, uniqueness margins
- **Servo** — pan/tilt gain, max angle, PWM center/range (1500 ±500 µs)
- **Display** — window name, box colors

## Dependencies

| Package | Required for |
|---|---|
| `opencv-python` | Capture, display, image processing |
| `ultralytics` | YOLO inference |
| `pymavlink` | MAVLink servo commands (optional — mock mode without it) |
| `pygrabber` | Camera auto-detection (optional — falls back to index 0) |
