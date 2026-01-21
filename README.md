# Animatron

Webcam-based visual effects application with real-time BPM detection. Apply various visual effects to your webcam feed that react to audio input.

## Features

- Real-time webcam visual effects
- BPM detection from microphone input
- Multiple effects including:
  - Spooky filter
  - Face replacement
  - Edge detection (with pulse and afterimage variants)
  - Matrix effect
  - ASCII webcam effect
  - Rhythmic light trails
  - Smiley face detection
  - Colorful circles
  - Circle grid effect

## Requirements

- Python 3.10+
- Poetry
- Webcam
- Microphone (for BPM detection)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd animatron
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

   This will create a virtual environment in the `.venv` folder within the project.

## Usage

Run the application:

```bash
poetry run python main.py
```

Or activate the virtual environment first:

```bash
poetry shell
python main.py
```

## Controls

- **N** - Switch to the next effect
- **X** or **ESC** - Exit the application

## Dependencies

- `opencv-python` - Webcam capture and image processing
- `numpy` - Numerical operations
- `pygame` - Display and window management
- `pyaudio` - Audio capture for BPM detection
- `librosa` - Audio analysis and BPM estimation
