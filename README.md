# AI-Driven Attention and Distraction Detector

A desktop computer-vision application that uses a webcam to monitor attention in real time. It detects whether the visible user is **Attentive**, **Distracted**, **Drowsy**, or **Unknown**, and generates a report for each monitoring session.

## Features

- Live webcam monitoring
- Face landmark detection with MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Head-position based distraction detection
- Short calibration period for user-specific eye thresholds
- CustomTkinter desktop dashboard
- Session history with one-second status logs
- Summary duration for every attention state
- Pie-chart report generation
- Report viewing and deletion from the application

## How it works

1. The webcam captures video frames using OpenCV.
2. MediaPipe Face Mesh identifies facial landmarks.
3. Six landmarks around each eye are used to calculate the Eye Aspect Ratio.
4. A two-second calibration estimates the user's normal open-eye value.
5. Eyes remaining below the calibrated threshold indicate drowsiness.
6. Horizontal movement of the nose landmark outside the centre range indicates distraction.
7. Time filters prevent ordinary blinks or momentary head movements from triggering false alerts.
8. The application records the status and creates a report when detection stops.

## Detection states

| State | Meaning |
|---|---|
| `Attentive` | Face is visible, eyes are open, and head is centred |
| `Distracted` | Head remains turned away from the centre |
| `Drowsy` | Eyes remain closed longer than the blink filter |
| `Unknown` | A reliable face/status cannot be determined |

## Technology stack

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- CustomTkinter
- Pillow
- Matplotlib

## Project structure

```text
.
├── app.py           # Main desktop application and report dashboard
├── detector.py      # Simpler standalone detection prototype
├── requirements.txt # Python dependencies
└── README.md
```

The `reports/` directory is created automatically when the application runs. Each session is stored in a timestamped folder containing:

- `log.csv` — status recorded during the session
- `summary.txt` — duration spent in each state
- `meta.json` — machine-readable session summary
- `pie.png` — attention distribution chart

## Installation

Python 3.10 or 3.11 is recommended for MediaPipe compatibility.

### 1. Download the repository

Use Git or download the repository as a ZIP from GitHub.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the desktop application

```bash
python app.py
```

Allow camera access when requested. Keep your face visible and look toward the camera during the initial calibration.

## Detection parameters

- Calibration time: approximately 2 seconds
- Drowsiness filter: approximately 1.2 seconds
- Distraction filter: approximately 1 second
- Horizontal centre tolerance: approximately ±0.15 from the frame centre
- Drowsiness threshold: calibrated open-eye EAR × 0.75, limited to the range 0.18–0.32

## Limitations

- Performance depends on lighting, camera angle, and face visibility.
- Glasses, occlusion, or unusual head posture can affect landmark detection.
- The project monitors one visible face and is intended as an educational prototype, not a medical or surveillance system.

## Privacy

Processing is performed locally. The application does not upload webcam frames to a remote service. Generated session reports remain in the local `reports/` directory.
