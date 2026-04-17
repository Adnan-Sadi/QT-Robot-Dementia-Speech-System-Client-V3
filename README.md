# QT Robot Dementia Speech System Client

This repository contains the **QT Robot client application** for the **Dementia Speech System** backend. This is a modified version of the [QT Robot Agentic Speech System Client](https://github.com/Adnan-Sadi/QT_Robot_Agentic_Speech_System_Client).

It provides a desktop UI for the QT robot operator and connects the robot to the backend conversational system. The application uses:

- **Google Cloud Speech-to-Text** for live speech recognition
- **WebSocket communication** with the backend
- **QT Robot ROS services** for speech, gestures, and emotions
- **CustomTkinter** for the desktop UI

This version uses **user-controlled turn-taking**:

1. The robot starts listening.
2. The user speaks freely.
3. The speech transcript is accumulated live in the UI.
4. The user clicks **Send** when they are finished speaking.
5. The transcript is sent to the backend.
6. The robot speaks the backend response.
7. Once the robot finishes speaking, it starts listening again.

This matches the turn-taking idea used in the frontend voice interaction of the main Dementia Speech System.

---

## Features

- **Continuous listening** with accumulated transcript
- **Manual send button** for user-controlled turn-taking
- **Improved UI** using CustomTkinter
- **Backend integration** through authenticated WebSocket communication
- **QT Robot speech and gesture output**
- **Modular architecture** for future backend-controlled robot actions such as:
  - gesture commands
  - facial emotion commands
  - movement commands

---

## Project Structure

```text
QT_Robot_Dementia_Speech_System_Client/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   └── settings.py              # Loads environment variables
│
├── controllers/
│   └── chat_controller.py       # Main turn-taking orchestration
│
├── services/
│   ├── audio_stream.py          # Audio queue stream helper
│   ├── backend_client.py        # Backend auth + WebSocket client
│   ├── event_bus.py             # UI/service event communication
│   ├── robot_actions.py         # QT Robot speech / gesture / emotion helpers
│   └── stt_accumulator.py       # Continuous Google STT with transcript accumulation
│
└── ui/
    ├── app.py                   # Main application window
    └── widgets/                 # Reusable UI widgets
```

---

## How It Works

### High-level flow

1. `main.py` starts the application.
2. ROS services for QT Robot are initialized.
3. The UI opens.
4. When the user starts a session:
   - the app connects to the backend
   - the speech recognizer starts listening
5. The recognizer continuously publishes interim and final transcript chunks to the UI.
6. When the user clicks **Send**:
   - the accumulated transcript is sent to the backend
   - listening is paused
   - the backend responds 
7. The robot speaks the response.
8. Listening resumes automatically.

---

## Requirements

### Python version
This application is intended to run on the QT robot environment using:

- **Python 3.8.10**

### System/runtime requirements
This app also assumes the following are available in the runtime environment:

- **ROS**
- QT Robot ROS services
- microphone audio topic:
  - `/qt_respeaker_app/channel0`
- QT Robot service endpoints such as:
  - `/qt_robot/speech/config`
  - `/qt_robot/behavior/talkText`
  - `/qt_robot/emotion/show`
  - `/qt_robot/gesture/play`

### Python packages
Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the repository root. You can copy the `.env.example` file and fill in the required values. The application uses these environment variables for configuration such as backend connection, speech recognition settings, and robot behavior parameters.

### Variable descriptions

| Variable | Description |
|---|---|
| `BASE_HTTP_URL` | Base URL of the Dementia Speech System backend |
| `WS_PATH` | WebSocket path used by the backend |
| `SOURCE` | Source label sent to the backend |
| `USERNAME` | Backend login username |
| `PASSWORD` | Backend login password |
| `MIC_SOURCE` | Audio source for the microphone input, default (QT Robot's microphone) or external |
| `MIC_DEVICE_INDEX` | Audio device index for the external microphone input, used by pyaudio (more instructions are given below for finding the device index) |
| `STT_ENGINE` | Speech-to-text engine; currently expected to be `gspeech` |
| `AUDIO_RATE` | Audio sample rate |
| `DEFAULT_LANGUAGE` | Default speech recognition language |
| `SPEECH_MODEL` | Google Speech model |
| `USE_ENHANCED_MODEL` | Whether to use enhanced Google STT model |
| `SPEECH_SPEED` | QT Robot speech speed |
| `DEFAULT_TIMEOUT` | STT stream timeout |
| `LLM_TIMEOUT` | Backend response timeout |
| `EMOTION_LISTENING` | Comma-separated list of QT listening emotions |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Adnan-Sadi/QT-Robot-Dementia-Speech-System-Client-V3.git
cd QT-Robot-Dementia-Speech-System-Client-V3
```

### 2. Create and activate a Python virtual environment

If you already use a robot-side Python virtual environment, activate that instead.

Example:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create the `.env` file

Create a `.env` file in the project root and fill in your backend credentials and settings.

### 5. Make sure ROS and QT Robot services are running

Before starting this application, confirm that:

- ROS is running
- QT robot services are available
- the microphone topic is publishing audio
- the backend is reachable from the robot


### 6. Find External Microphone Device Index (if using external mic)
---
Find if the device is available:

```bash
arecord -l
```

Find the device index for PyAudio configuration:

```bash
python3 << 'EOF'
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Index {i}: {info['name']} (inputs: {info['maxInputChannels']}, rate: {int(info['defaultSampleRate'])})")
p.terminate()
EOF
```

set the `MIC_DEVICE_INDEX` variable in the `.env` file.

## Running the Application

Start the app with:

```bash
python3 main.py
```

When the app launches:

1. Click **Start Chat**
2. Speak to the robot
3. Watch the transcript accumulate in the UI
4. Click **Send** when you finish speaking
5. Wait for the robot to respond
6. After the robot finishes speaking, it will return to listening mode

To stop the session, click **Stop Chat**.