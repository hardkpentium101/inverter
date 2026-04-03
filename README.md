# Hand Landmark Inverter

A real-time hand landmark detection application using MediaPipe and OpenCV that inverts the video feed between detected hand landmarks (thumb and index finger tips).

## Features

- Real-time hand detection using MediaPipe's Hand Landmarker
- Tracks thumb (landmark 4) and index finger (landmark 8) tips for both hands
- Inverts the video region between detected hand points
- Displays live camera feed with overlay

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MediaPipe hand landmarker model:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

Run the application:

```bash
python app.py
```

A window will open showing your camera feed. The app tracks your hand landmarks and inverts the region between your thumb and index finger tips. Press `q` to quit.

## Project Structure

```
.
├── app.py                    # Main application
├── hand_landmarker.task      # MediaPipe hand landmarker model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dependencies

- **opencv-python**: Computer vision operations and camera capture
- **mediapipe**: Hand landmark detection
- **pprintpp**: Enhanced pretty-printing (utility)

## How It Works

1. Captures video from the default webcam
2. Converts frames to RGB for MediaPipe processing
3. Detects hand landmarks (up to 2 hands)
4. Extracts thumb and index finger tip coordinates
5. Draws lines between landmarks and inverts the pixel values in the enclosed region
6. Displays the processed feed in real-time
