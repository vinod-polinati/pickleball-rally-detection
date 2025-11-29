# 🎾 Pickleball Rally Detection & Splitter

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Automatically detect, segment, and extract individual rally clips from pickleball match videos using computer vision and deep learning.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Technical Details](#-technical-details)
- [Example Output](#-example-output)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project automatically analyzes pickleball/badminton match videos to:

1. **Detect** the ball in every frame using YOLOv8
2. **Track** ball movement with physics-based validation
3. **Identify** rally boundaries using intelligent gap detection
4. **Extract** individual rally clips with precise timestamps

Perfect for coaches, players, and analysts who want to break down match footage into digestible rally segments.

## ✨ Features

### Core Capabilities

- AI-Powered Detection\*\*: Uses YOLOv8 for robust ball and player detection
- Smart Filtering\*\*: Multi-stage validation removes false positives
- Intelligent Segmentation\*\*: Detects rally start/end with physics-based cuts
- Fast Processing\*\*: FFmpeg stream copying for quick video extraction
- Debug Mode\*\*: Visual overlay showing detection process

### Unique Innovations

- **Shoe Filter**: Eliminates false ball detections from players' shoes
- **Teleport Detection**: Identifies scene cuts via impossible ball movements
- **Gap Tolerance**: Bridges brief occlusions without splitting rallies
- **Physics Validation**: Rejects detections that violate motion constraints

## 🔍 How It Works

```
┌─────────────────┐
│  Input Video    │
│  (Full Match)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         YOLO Object Detection           │
│  • Detects balls (class 32)             │
│  • Detects players (class 0)            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Multi-Stage Filtering           │
│  1. Size Filter (max 65x65px)           │
│  2. Shoe Filter (foot zone detection)   │
│  3. Physics Filter (max 300px jump)     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Timeline Construction           │
│  • Binary array: 1=ball, 0=no ball      │
│  • Mark forced cuts (teleports)         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Rally Segmentation              │
│  • Apply 0.6s gap tolerance             │
│  • Respect forced cut boundaries        │
│  • Filter by minimum duration (1.0s)    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Video Extraction                │
│  • FFmpeg stream copy                   │
│  • Individual rally clips               │
│  • rally_01.mp4, rally_02.mp4, ...      │
└─────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed and accessible in PATH
- CUDA-capable GPU (optional, but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/vinod-polinati/pickleball-rally-detection.git
cd pickleball-rally-detection
```

### Step 2: Install Dependencies

```bash
pip install opencv-python numpy ultralytics
```

### Step 3: Install FFmpeg

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Step 4: Download YOLO Model

The YOLOv8x model will download automatically on first run, or manually:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

## 🚀 Usage

### Basic Usage

```bash
python rally_detector.py
```

Place your match video as `input_match.mp4` in the project directory. The script will create a `rallies_v18/` folder with individual clips.

### Debug Mode

First run with debug enabled to verify detection quality:

```python
# In rally_detector.py
DEBUG_MODE = True
DEBUG_DURATION_LIMIT = 40  # Process first 40 seconds
```

Run the script:

```bash
python rally_detector.py
```

Check `debug_v18.mp4` to see:

- 🟢 Green circles on detected balls
- 🟠 Orange boxes on filtered shoes
- 🔴 Red borders on scene cuts/teleports

### Production Mode

After verifying debug output:

```python
DEBUG_MODE = False
```

Run full video processing:

```bash
python rally_detector.py
```

## ⚙️ Configuration

### Key Parameters

| Parameter            | Default             | Description                           |
| -------------------- | ------------------- | ------------------------------------- |
| `VIDEO_PATH`         | `"input_match.mp4"` | Input video file path                 |
| `OUTPUT_DIR`         | `"rallies_v18"`     | Output directory for clips            |
| `GAP_TOLERANCE_SEC`  | `0.6`               | Max gap before splitting rally        |
| `MAX_BALL_JUMP`      | `300.0`             | Max pixels/frame (teleport threshold) |
| `FOOT_ZONE_RATIO`    | `0.45`              | Bottom % of player = foot zone        |
| `MIN_RALLY_DURATION` | `1.0`               | Minimum seconds for valid rally       |
| `CONF_THRESHOLD`     | `0.15`              | YOLO confidence threshold             |
| `IMG_SIZE`           | `1280`              | YOLO inference image size             |

### Tuning Guide

**Too many false rallies?**

- Increase `MIN_RALLY_DURATION` (try 1.5 or 2.0)
- Increase `CONF_THRESHOLD` (try 0.20 or 0.25)

**Rallies merging incorrectly?**

- Decrease `GAP_TOLERANCE_SEC` (try 0.4 or 0.5)
- Decrease `MAX_BALL_JUMP` (try 250)

**Missing valid rallies?**

- Decrease `CONF_THRESHOLD` (try 0.10)
- Increase `GAP_TOLERANCE_SEC` (try 0.8 or 1.0)

**Shoe false positives?**

- Increase `FOOT_ZONE_RATIO` (try 0.50 or 0.55)

## 🔬 Technical Details

### Detection Pipeline

#### 1. Object Detection

- **Model**: YOLOv8x (extra-large variant)
- **Classes**: Person (0), Sports Ball (32)
- **Confidence**: 15% minimum
- **Image Size**: 1280px (balances speed/accuracy)

#### 2. Size Filtering

Rejects ball detections larger than 65x65 pixels to eliminate:

- Net posts
- Court equipment
- Player torsos
- Background objects

#### 3. Shoe Filtering

```python
def is_shoe(ball_box, person_boxes):
    # Checks if ball is in bottom 45% of player bounding box
    # + 40px buffer below feet
```

Eliminates the most common false positive in court sports.

#### 4. Physics Validation

```python
MAX_BALL_JUMP = 300.0  # pixels
```

Real ball movements:

- Smash: ~100px/frame
- Regular shot: ~50px/frame
- Scene cut: ~800px/frame

Any movement >300px triggers a "forced cut" to prevent rally merging.

#### 5. Rally Segmentation

```python
GAP_TOLERANCE = 0.6s  # 18 frames @ 30fps
```

Allows brief occlusions (ball behind player, net, etc.) without splitting rallies.

## 📊 Example Output

### Input

```
input_match.mp4 (45:30 duration)
```

### Output

```
rallies_v18/
├── rally_01.mp4  (0:08 - First serve rally)
├── rally_02.mp4  (0:12 - Long baseline exchange)
├── rally_03.mp4  (0:05 - Quick net point)
├── rally_04.mp4  (0:15 - Extended rally)
...
└── rally_87.mp4  (0:09 - Final point)
```

### Console Output

```
🚀 Starting V18 (Goldilocks Tune)...
Info: 1920x1080 @ 30.0FPS
Scanning: 81450/81450

Slicing Timeline...
Final Count: 87 rallies found.
Done.
```

## 🐛 Troubleshooting

### Issue: "YOLO model not found"

**Solution:**

```bash
pip install ultralytics --upgrade
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

### Issue: "FFmpeg not recognized"

**Solution:** Add FFmpeg to system PATH or specify full path:

```python
"C:/ffmpeg/bin/ffmpeg.exe"  # Windows
"/usr/local/bin/ffmpeg"      # macOS/Linux
```

### Issue: Too many false detections

**Solution:** Enable debug mode and check:

1. Are shoes being filtered? (should show orange boxes)
2. Is confidence too low? (increase `CONF_THRESHOLD`)
3. Are cuts being detected? (should show red borders)

### Issue: CUDA out of memory

**Solution:** Reduce `IMG_SIZE`:

```python
IMG_SIZE = 640  # or 960
```

### Issue: Missing rallies

**Solution:**

1. Check debug video for ball visibility
2. Lower `CONF_THRESHOLD` to 0.10
3. Increase `GAP_TOLERANCE_SEC` to 1.0
4. Verify input video quality/lighting

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add support for multiple camera angles
- [ ] Implement player tracking for advanced analytics
- [ ] Create GUI for parameter tuning
- [ ] Add ball trajectory visualization
- [ ] Support for real-time processing
- [ ] Export rally statistics (duration, ball speed, etc.)

### Development Setup

```bash
git checkout -b feature/your-feature
# Make changes
python rally_detector.py  # Test
git commit -am "Add feature"
git push origin feature/your-feature
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **OpenCV** for computer vision primitives
- **FFmpeg** for video processing
- Inspired by sports analytics community

---

⭐ **Star this repo** if it helped your pickleball analysis!  
🐛 **Report issues** on the [Issues page](https://github.com/vinod-polinati/pickleball-rally-detection/issues)  
💬 **Discuss** in [Discussions](https://github.com/vinod-polinati/pickleball-rally-detection/discussions)
