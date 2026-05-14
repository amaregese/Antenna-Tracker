# AI Antenna Tracker — Technical Report

## 1. Project Overview

The AI Antenna Tracker is a real-time computer vision system that autonomously detects, selects, and tracks moving objects with a pan/tilt antenna mount. It uses a webcam and YOLO object detection to compute the target's position, then drives a Pixhawk-controlled servo gimbal via MAVLink to keep the antenna continuously aimed at the object.

The system runs at approximately 15–30 FPS on consumer hardware and supports automatic re-acquisition if the target is temporarily lost.

---

## 2. System Architecture

```
                     ┌─────────────────┐
                     │    Camera       │
                     │ (DirectShow)    │
                     └────────┬────────┘
                              │ frames
                              ▼
                     ┌─────────────────┐
                     │  YOLO11n        │
                     │  (Ultralytics)  │
                     │  Object Detection│
                     └────────┬────────┘
                              │ detections
                              ▼
               ┌─────────────────────────┐
               │  Multi-Modal Feature    │
               │  Extraction & Matching  │
               │  (Color, Shape, Edge,   │
               │   Gradient, Contour)    │
               └────────┬────────────────┘
                        │ selected object
                        ▼
               ┌─────────────────────────┐
               │  Pan/Tilt Calculation   │
               │  (Proportional Control) │
               └────────┬────────────────┘
                        │ angles
                        ▼
               ┌─────────────────────────┐
               │  MAVLink DO_SET_SERVO   │
               │  → Pixhawk (USB Serial) │
               │  ch1: TrackerPitch      │
               │  ch2: TrackerYaw        │
               └─────────────────────────┘
```

### Data Flow

1. **Capture** — 640x480 frames from webcam via OpenCV DirectShow
2. **Detection** — YOLO11n runs inference at 480px resolution; filters by confidence (≥0.3) and minimum area (0.05% of frame)
3. **User Selection** — click an object; system extracts ~20 feature descriptors for that object
4. **Frame-to-Frame Tracking** — each new frame, same-class detections are scored against the stored reference; best unique match above threshold is selected
5. **Re-Acquisition** — if tracking is lost, a secondary matching stage searches within a velocity-predicted window for up to 90 frames using richer multi-modal features
6. **Pan/Tilt Control** — proportional mapping from pixel offset to servo angle (clamped ±45°)
7. **Servo Output** — MAVLink `DO_SET_SERVO` commands over USB serial to Pixhawk

---

## 3. Hardware Setup

### Components

| Component | Model / Specification |
|---|---|
| Camera | Built-in laptop webcam (640×480) |
| Flight Controller | Pixhawk (Pixhawk 1 / compatible) |
| Connection | USB Serial (COM5, 57600 baud) |
| Servo 1 (Ch1) | TrackerPitch (SERVO1_FUNCTION=72) — elevation |
| Servo 2 (Ch2) | TrackerYaw (SERVO2_FUNCTION=71) — azimuth |
| Computer | Windows 11, Python 3.12 |

### Pixhawk Configuration

The Pixhawk is configured via its parameter system:

| Parameter | Value | Function |
|---|---|---|
| `SERVO1_FUNCTION` | 72 | TrackerPitch — controls antenna elevation |
| `SERVO2_FUNCTION` | 71 | TrackerYaw — controls antenna azimuth |

Servo PWM range: 1000–2000 µs with 1500 µs as center (neutral position).

---

## 4. Software Pipeline — Detailed

### 4.1 Camera Module (`vision/camera.py`)

- Auto-detects physical cameras using DirectShow enumeration (via `pygrabber`)
- Skips virtual devices (NDI, OBS virtual camera)
- Falls back to camera index 0 if enumeration fails
- Resolution forced to 640×480 via config

### 4.2 Object Detection (`vision/detector.py`)

- Model: **YOLO11n** (nano, ~5.6M parameters) from Ultralytics
- 80 trained classes (COCO dataset)
- Post-processing: Non-maximum suppression (IoU threshold 0.45), minimum confidence filter (0.3), minimum area filter
- Frame-skip capability configurable in `config.py`

### 4.3 Feature Extraction (`vision/matching.py`)

Upon selecting an object, the system extracts a comprehensive set of visual descriptors:

| Feature Category | Method | Dimensionality |
|---|---|---|
| 1D Color Histogram | HSV Hue, 180 bins | 180 values |
| 2D Color Histogram | HSV Hue+Saturation, 64×64 bins | 4096 values |
| 3D Color Histogram | HSV full, 32×32×32 bins | 32768 values |
| Regional Histograms | Top/bottom/left/right quadrants, 64 bins each | 4 × 64 values |
| Mean & Std HSV | Average and spread of H, S, V | 6 values |
| Hu Moments | 7 invariant moments (log-scaled) | 7 values |
| Contour Features | Solidity, extent, circularity, vertex count, area | 5 values |
| Edge Density | Canny edge fraction in ROI | 1 value |
| Gradient Stats | Sobel magnitude mean & std | 2 values |
| Geometry | Area, aspect ratio, width, height, center | 5 values |

### 4.4 Tracking Strategy

**Active Tracking** (object visible in current frame):
- Weighted scoring: IOU (30 pts) + area ratio (10) + aspect ratio (5) + center proximity (10) + histogram correlation (5) + confidence (5) + base (40)
- Uniqueness requirement: best score must exceed second-best by a margin (default 7 pts)
- Minimum threshold: score > 45 to consider a match valid

**Lost Target Re-Acquisition** (object not found):
- 90-frame memory (≈3–6 seconds)
- Velocity prediction for first 30 frames to estimate expected position
- Richer scoring: Hu moments (15 pts) + contour features (15) + 3D histogram (25) + regional layout (20) + shape consistency (15) + edge/gradient (10) + center proximity (5) + confidence (5)
- Area similarity floor: candidate must be ≥35% of reference area
- Position constraint: candidate must be within 60% of frame diagonal from predicted position

### 4.5 Control Law (`main.py`)

```python
x_delta = (object_center_x - frame_center_x) / frame_width
y_delta = (object_center_y - frame_center_y) / frame_height

pan  = clamp(x_delta * MAX_PAN * GAIN_PAN)     # ±45° max
tilt = clamp(y_delta * MAX_TILT * GAIN_TILT)   # ±45° max
```

- **GAIN_PAN = GAIN_TILT = 2.0** — proportional amplification
- **MAX_PAN = MAX_TILT = 45°** — software limit
- PWM mapping: `1500 + (angle / 90) × 500` µs (range 1000–2000 µs)

For Pixhawk compatibility:
- Pan (azimuth) sent to channel 2 (TrackerYaw)
- Tilt (elevation) sent to channel 1 (TrackerPitch), with sign inverted (Pixhawk convention: positive pitch = UP; camera y-axis positive = DOWN)

### 4.6 User Interface (`ui/visualizer.py`)

Two OpenCV windows:

1. **Antenna Tracker** — camera feed with bounding boxes, center crosshair, green line to tracked object, FPS/pan/tilt overlay
2. **Tracker Scope** — 540×620 diagnostic panel containing:
   - Direction arrows with color-coded pan/tilt readouts
   - Target class, confidence, tracking status
   - Radar scope with cardinal directions, range rings, pulsing target dot
   - Horizontal gauge bars for pan and tilt with numerical values
   - Servo position indicator crosshair
   - Status bar (CENTERED/TRACKING)

### 4.7 Hardware Communication (`hardware/tracker.py`)

- Uses `pymavlink` to send MAVLink `MAV_CMD_DO_SET_SERVO` commands
- Connection: USB serial with auto-reconnect
- Heartbeat timeout: 10 seconds
- Print all commands in mock mode (no hardware connected)

---

## 5. Configuration Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.3 | Minimum YOLO detection confidence |
| `IOU_THRESHOLD` | 0.45 | NMS overlap threshold |
| `MIN_BOX_AREA_RATIO` | 0.0005 | Minimum detection area (fraction of frame) |
| `INFERENCE_IMG_SIZE` | 480 | YOLO input resolution |
| `TRACK_MATCH_THRESHOLD` | 45.0 | Minimum score to maintain tracking |
| `REACQUIRE_MATCH_THRESHOLD` | 60.0 | Minimum score to re-acquire lost target |
| `TARGET_MEMORY_FRAMES` | 90 | Frames to remember a lost target |
| `GAIN_PAN / GAIN_TILT` | 2.0 | Proportional control gain |
| `MAX_PAN / MAX_TILT` | 45.0 | Maximum servo deflection (degrees) |
| `SERVO_CENTER` | 1500 | Neutral PWM (µs) |
| `SERVO_RANGE` | 500 | ±PWM range from center (µs) |
| `FRAME_SKIP` | 1 | Run detection every N frames |

---

## 6. Performance Characteristics

| Metric | Value |
|---|---|
| Detection Model | YOLO11n (COCO, 80 classes) |
| Frame Rate | ~15–30 FPS (CPU-dependent) |
| Tracking Latency | <100 ms (frame-to-frame) |
| Pan/Tilt Range | ±45° software-limited |
| Servo Resolution | ~0.18° per µs PWM step |
| Re-Acquisition Window | ~3–6 seconds (90 frames) |
| Power (software) | <5% CPU (modern laptop) |

---

## 7. Usage Instructions

### Basic Run
```bash
python main.py --port COM5 --baud 57600
```

### Command Line Arguments

| Argument | Default | Purpose |
|---|---|---|
| `--camera-index` | auto | Camera device index |
| `--conf-threshold` | 0.3 | Detection confidence cutoff |
| `--model` | models/yolo11n.pt | YOLO model path |
| `--port` | COM3 | Pixhawk serial port |
| `--baud` | 57600 | Serial baud rate |
| `--pan-channel` | 2 | Servo channel for pan (TrackerYaw) |
| `--tilt-channel` | 1 | Servo channel for tilt (TrackerPitch) |
| `--no-pixhawk-mode` | false | Disable tilt sign inversion |

### Controls
- **Left-click** on a detected object to start tracking
- **C** key to clear current selection
- **Q** key to quit

---

## 8. Project Structure

```
Antenna-Tracker/
├── main.py                    # Entry point, control loop
├── config.py                  # All tunable parameters
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
├── REPORT.md                  # This report
│
├── vision/
│   ├── camera.py              # DirectShow camera capture
│   ├── detector.py            # YOLO detection + tracking
│   └── matching.py            # Feature extraction + scoring
│
├── hardware/
│   └── tracker.py             # MAVLink servo interface
│
├── ui/
│   └── visualizer.py          # Diagnostic display
│
└── models/
    └── yolo11n.pt             # YOLO11 nano weights
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥4.x | Image capture, processing, display |
| `ultralytics` | ≥8.x | YOLO object detection |
| `pymavlink` | ≥2.x | MAVLink protocol for Pixhawk communication |
| `pyserial` | ≥3.x | USB serial connection to Pixhawk |
| `pygrabber` | ≥0.3 | DirectShow camera enumeration |

---

## 10. Hardware Requirements

| Component | Status | Notes |
|---|---|---|
| Camera (USB webcam) | **Required** | Any USB camera (Logitech C920 or similar). Jetson Nano supports USB cameras via V4L2. Minimum 640×480 resolution. |
| Pixhawk | Required | Connected via USB-serial (micro USB cable) |
| Servos (×2) | Required | Standard RC servos for pan/tilt |
| Computer | Required | Currently running on Windows laptop |


---

## 11. Future Improvements

- **Jetson Nano deployment** — port the system to run on Jetson Nano (ARM64 / aarch64). Jetson provides GPU acceleration via CUDA which can significantly improve YOLO inference speed. Requires:
  - Ubuntu 20.04 L4T (JetPack SDK)
  - Ultralytics with PyTorch compiled for aarch64 / CUDA
  - USB or CSI camera connection
  - Serial (UART) or USB connection to Pixhawk
  - Potential frame size increase to 1280×720 with GPU acceleration
