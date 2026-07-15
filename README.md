# QT Robot Dementia Speech System Client

This repository contains the **QT Robot client application** for the **Dementia Speech System** backend. This is a modified version of the [QT Robot Agentic Speech System Client](https://github.com/Adnan-Sadi/QT-Robot-Agentic-Speech-System-Client).

It provides a desktop UI for the QT robot operator and connects the robot to a cloud-based LLM conversational backend. The application uses:

- **WebSocket communication** with the backend (audio streamed directly — backend handles STT)
- **QT Robot ROS services** for speech, gestures, and emotions
- **CustomTkinter** for the desktop UI

### Turn-taking flow

1. The operator clicks **Start Chat** — the robot greets the user and starts listening.
2. The user speaks freely.
3. The operator clicks **Send** when the user has finished speaking.
4. The accumulated audio is sent to the backend, which transcribes it and generates a response.
5. The robot speaks the response with a matching gesture.
6. Once the robot finishes speaking, it automatically resumes listening.
7. When the backend signals the conversation is complete (`close_session`), the application closes automatically after a short countdown.

---

## Features

- **Audio streaming to backend** — raw PCM audio is buffered locally and sent to the backend on Send; the backend handles speech-to-text
- **Manual send button** for user-controlled turn-taking
- **Always-visible settings panel** on the left side of the main window — no hidden menus
- **Live-adjustable settings** — speech speed, volume, and transcript font size apply immediately without needing to click Apply
- **Microphone management** — supports both QT Robot's built-in ReSpeaker mic and external USB microphones, with an Apply button that only restarts the mic stream when needed
- **Persistent user settings** — last-used speed, volume, font size, and microphone are saved to a local `user_settings.json` file and restored on the next launch
- **Session auto-close** — when the backend sends `close_session: true`, the robot finishes speaking, then the application closes with a countdown
- **Modular architecture** for future backend-controlled robot actions (gestures, emotions, movement)

---

## Project Structure

```text
QT-Robot-Dementia-Speech-System-Client-V3/
├── main.py                          # Entry point
├── launch.sh                        # Double-clickable desktop launcher
├── requirements.txt
├── README.md
├── .env                             # Your local config (gitignored)
├── .env.example                     # Template for .env
├── user_settings.json               # Auto-generated; persists UI settings (gitignored)
│
├── config/
│   ├── settings.py                  # Loads environment variables into a Settings object
│   └── user_settings.py             # Loads/saves user_settings.json; resolves mic index by name
│
├── controllers/
│   └── chat_controller.py           # Orchestrates the turn-taking session lifecycle
│
├── services/
│   ├── backend_client.py            # Backend auth + WebSocket client (BackendBridge)
│   ├── event_bus.py                 # Thread-safe UI/service event queue
│   ├── robot_actions.py             # QT Robot speech / gesture / emotion ROS wrappers
│   └── stt_accumulator.py           # Captures raw PCM audio; resamples for backend
│
└── ui/
    ├── app.py                       # Main application window (MainWindow)
    └── widgets/
        ├── settings_panel.py        # Left-side settings panel (always visible)
        ├── transcript_panel.py      # Scrollable robot response transcript
        └── status_bar.py            # Bottom status bar
```

---

## Requirements

### Python version

- **Python 3.8.10** (QTRobotV2's default)

### System requirements

The following must be available in the runtime environment on the robot:

- **ROS Noetic** (or compatible)
- QT Robot ROS services running, specifically:
  - `/qt_robot/speech/say`
  - `/qt_robot/speech/config`
  - `/qt_robot/behavior/talkText`
  - `/qt_robot/emotion/show`
  - `/qt_robot/gesture/play`
  - `/qt_robot/setting/setVolume`
- Microphone audio topic (if using built-in mic):
  - `/qt_respeaker_app/channel0`

---

## Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `BASE_HTTP_URL` | ✅ | Base URL of the Dementia Speech System backend |
| `WS_PATH` | ✅ | WebSocket path on the backend |
| `SOURCE` | ✅ | Source label sent to the backend |
| `USERNAME` | ✅ | Backend login username |
| `PASSWORD` | ✅ | Backend login password |
| `AUDIO_RATE` | | Audio sample rate in Hz (default: `16000`) |
| `MIC_SOURCE` | | `default` for QT built-in ReSpeaker mic, `external` for USB mic (default: `default`) |
| `MIC_DEVICE_INDEX` | | PyAudio device index for external mic — only needed as a starting point; the app resolves by name automatically after first use |
| `SPEECH_SPEED` | | Robot speech speed (default: `90`) |
| `SPEECH_VOLUME` | | Robot speaker volume 0–100 (default: `80`) |
| `GREETING_TEXT` | | Text spoken at the start of each session |
| `LLM_TIMEOUT` | | Backend response timeout in seconds (default: `25.0`) |
| `EMOTION_LISTENING` | | Comma-separated QT emotion names shown while listening |

> **Note:** `SPEECH_SPEED`, `SPEECH_VOLUME`, and microphone settings can all be changed at runtime from the Settings panel in the UI and will be saved automatically. The `.env` values serve as the initial defaults only.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Adnan-Sadi/QT-Robot-Dementia-Speech-System-Client-V3.git
cd QT-Robot-Dementia-Speech-System-Client-V3
```

### 2. Create the virtual environment

```bash
python3 -m venv dss_venv
source dss_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create the `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your backend URL, credentials, and any other values you want to override.

### 5. (Optional: kept from old version of app) Find the external USB microphone device index

If you are using an external USB microphone (such as the JOUNIVO USB Microphone), you can find its device index to set as a starting point in `.env`. After the first time you click **Apply Microphone** in the UI, the device name is saved and the index is resolved automatically on future launches — you will not need to update this manually again.

Check if the device is visible to the OS:

```bash
arecord -l
```

Find the PyAudio device index:

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

Set `MIC_DEVICE_INDEX` in your `.env` to the index shown for your microphone.

---

## Making the Application Double-Clickable (One-Time Setup)

These steps only need to be done once. After this, the application can be launched by double-clicking an icon on the robot's desktop.

### Step 1: Make the launcher script executable

In a terminal on the robot:

```bash
chmod +x /home/qtrobot/catkin_ws/src/qt_dss_app/src/QT-Robot-Dementia-Speech-System-Client-V3/launch.sh
```

Replace `/path/to/` with the actual path to the cloned repository (e.g. `/home/qtrobot/catkin_ws/src/qt_dss_app/src/`). 

### Step 2: Create a desktop shortcut file

Create a `.desktop` file so the file manager recognises it as a launchable application:

```bash
nano ~/Desktop/qt-speech-system.desktop
```

Paste the following, replacing the paths with the actual location of the repository on the robot:

```ini
[Desktop Entry]
Version=1.0
Type=Application
Name=QT Speech System
Comment=Launch the QT Robot Dementia Speech System Client
Exec=/home/qtrobot/catkin_ws/src/qt_dss_app/src/QT-Robot-Dementia-Speech-System-Client-V3/launch.sh
Icon=utilities-terminal
Terminal=true
Categories=Application;
```

> Set `Terminal=true` to keep the terminal window open while the app is running. This is useful because you can see status messages and errors. Set it to `false` if you want a cleaner experience once everything is confirmed working.

Save the file (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 3: Mark the desktop shortcut as trusted/executable

```bash
chmod +x ~/Desktop/qt-speech-system.desktop
```

On some desktop environments (such as LXDE, which QT robot uses), you may also need to right-click the file and select **"Trust this executable"** or **"Allow executing"** from the context menu.

### Step 4: Test it

Double-click the icon on the desktop. The terminal window should open, and you should see:

```
[Launcher] Dependencies up to date. Skipping install.
[Launcher] Starting QT Robot Speech System...
```

On the very first run, or after `requirements.txt` changes, you will instead see:

```
[Launcher] requirements.txt has changed (or first run). Installing dependencies...
...
[Launcher] Dependencies installed successfully.
[Launcher] Starting QT Robot Speech System...
```

---

## Running the Application Manually (Terminal)

If you prefer to run directly from a terminal:

```bash
cd /path/to/QT-Robot-Dementia-Speech-System-Client-V3
source dss_venv/bin/activate
source /opt/ros/noetic/setup.bash   # adjust for your ROS version
python3 main.py
```

---

## Using the Application

Once the window opens:

1. Adjust settings in the left panel if needed (microphone, speech speed, volume, font size)
2. Click **▶ Start Chat** — the robot will greet the user and begin listening
3. The user speaks
4. Click **Send** when the user finishes speaking
5. Wait for the robot to respond — the response appears in the transcript panel
6. The robot automatically resumes listening after responding
7. Click **■ Stop Chat** at any time to end the session manually
8. The session ends automatically when the backend signals the conversation is complete

### Settings panel

| Setting | Description |
|---|---|
| **Microphone** | Select the audio input device. Click **Apply Microphone** to activate the change (this restarts the audio stream) |
| **Speed** | Robot speech speed. Changes apply immediately |
| **Volume** | Robot speaker volume. Changes apply immediately |
| **Font size** | Transcript panel text size. Changes apply immediately |

All settings (except microphone) are applied live without needing to click any button. Settings are saved automatically and restored on the next launch.

---