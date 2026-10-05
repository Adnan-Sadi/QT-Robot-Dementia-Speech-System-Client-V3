# QT Robot Dementia Speech System Client

This repository contains the **QT Robot client application** for the **Dementia Speech System** backend. This is a modified version of the [QT Robot Speech System Client](https://github.com/Adnan-Sadi/QT-Robot-Agentic-Speech-System-Client).

It provides a desktop UI for the QT robot operator and connects the robot to a cloud-based LLM conversational backend. The application uses:

- **WebSocket communication** with the backend (audio streamed directly — backend handles STT)
- **QT Robot ROS services** for speech, gestures, and emotions
- **CustomTkinter** for the desktop UI

### Turn-taking flow

1. The operator clicks **Start Chat** — the robot greets the user and starts listening.
2. The user speaks freely.
3. The operator clicks **Send** or presses **Enter** on the main screen when the user has finished speaking.
4. Send finalizes the streamed audio turn; the backend completes transcription and generates a response.
5. The robot speaks the response with a matching gesture.
6. Once the robot finishes speaking, it automatically resumes listening.
7. When the backend signals the conversation is complete (`chat_ended`), the application closes automatically after a short countdown.

---

## Features

- **Audio streaming to backend** — microphone audio is streamed to the backend, which handles speech-to-text.
- **Simple conversation screen** — a large circular Send button is the main focus; settings and transcript text are hidden initially.
- **Separate settings screen** — open Settings from the toolbar and return using Back to conversation.
- **Keyboard shortcuts** — Enter sends from the main screen; arrow keys adjust volume and speech speed from either screen.
- **Try my voice** — test speed and volume before a conversation using a repeating local story.
- **Live speech adjustments** — adjust speed and volume using sliders, buttons, or keyboard shortcuts.
- **Collapsible transcript** — reveal the latest robot response when needed and adjust its text size directly in the transcript header.
- **Microphone management** — select the built-in ReSpeaker or an external microphone before starting a session.
- **Optional microphone recording** — save microphone input to timestamped FLAC files in `recordings/`.
- **Persistent preferences** — microphone selection, speed, volume, transcript text size, and recording preference are saved in `user_settings.json`.
- **Session auto-close** — after the final response, the application saves preferences and closes following a countdown.
- **Modular architecture** — separate services handle backend communication, robot actions, recording, and voice preview.

---

## Project Structure

```text
QT-Robot-Dementia-Speech-System-Client-V3/
├── main.py                          # Entry point
├── launch.sh                        # Double-clickable desktop launcher
├── requirements.txt
├── README.md
├── .env                             # Local configuration (gitignored)
├── .env.example                     # Environment configuration template
├── user_settings.json               # Auto-generated saved preferences
│
├── assets/
│   └── voice_preview_story.txt       # Editable story for voice setup
│
├── recordings/                      # Created when microphone recording starts
│
├── config/
│   ├── settings.py                  # Environment defaults and UI configuration
│   └── user_settings.py             # Preference persistence and mic resolution
│
├── controllers/
│   └── chat_controller.py           # Coordinates conversations and voice preview
│
├── services/
│   ├── backend_client.py            # Backend authentication and WebSocket client
│   ├── event_bus.py                 # Thread-safe UI/service event queue
│   ├── robot_actions.py             # Robot speech, gesture, and emotion wrappers
│   ├── session_recorder.py          # Records microphone input to FLAC
│   ├── stt_accumulator.py           # Captures and streams microphone audio
│   └── voice_preview.py             # Repeating story and preview cancellation
│
└── ui/
    ├── app.py                       # Conversation/settings navigation and shortcuts
    └── widgets/
        ├── settings_panel.py        # Scrollable settings and voice-preview controls
        ├── transcript_panel.py      # Latest response and compact text-size controls
        └── status_bar.py            # Status and error messages
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
| `MIC_DEVICE_INDEX` | | PyAudio device index for external mic. This can be set manually but is not needed with the new versions as the app resolves by name automatically after first use |
| `SPEECH_SPEED` | | Robot speech speed (default: `90`) |
| `SPEECH_VOLUME` | | Robot speaker volume 0–100 (default: `80`) |
| `TRANSCRIPT_FONT_SIZE` | | Initial transcript text size, 10–50 (default: `16`); saved preferences override this value |
| `GREETING_TEXT` | | Text spoken at the start of each session |
| `LLM_TIMEOUT` | | Backend response timeout in seconds (default: `25.0`) |
| `EMOTION_LISTENING` | | Comma-separated QT emotion names shown while listening |

---

## Setup Instructions

### 1. Create the new ROS Python project. (QT Robot only)

- Navigate to the `src` folder of your catkin workspace:
```bash
cd ~/catkin_ws/src 
```
- Create a new ROS package for the QT Robot Dementia Speech System Client:
```bash
catkin_create_pkg qt_dss_app std_msgs rospy roscpp -D "Dementia speech system application connected to the cloud backend" -V "1.0.0" -a "Adnan Sadi" 
```
> [!TIP]
> The text inside the quotes (`-D "..."`, `-V "..."`, and `-a "..."`) represents project metadata description, version, and author. You can customize these values to anything you want when creating your own ROS project. You can also customize the package name (`qt_dss_app`) to your liking, but make sure to update the paths in the following steps accordingly.

### 2. Clone the repository

- Navigate to the `src` folder of the newly created ROS project:
```bash
cd qt_dss_app/src 
```

- Clone this repository into the `src` folder:
```bash
git clone https://github.com/Adnan-Sadi/QT-Robot-Dementia-Speech-System-Client-V3.git
```
- Navigate into the cloned repository:
```bash
cd QT-Robot-Dementia-Speech-System-Client-V3
```

### 3. Create the virtual environment
- In case the venv package is missing, run: 
```bash
sudo apt install python3.8-venv 
```

- Create a virtual environment:

```bash
python3 -m venv dss_venv
```

- activate the virtual environment:

```bash
source dss_venv/bin/activate
```

### 4. Install dependencies and create the `.env` file

- Update pip (not strictly necessary):

```bash
pip install --upgrade pip
```
- Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

- Create the `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your backend URL, credentials, and any other values you want to override.


### 5. Making the Application Double-Clickable (One-Time Setup)

These steps only need to be done once. After this, the application can be launched by double-clicking an icon on the robot's desktop.

#### Step 1: Make the launcher script executable

In a terminal on the robot:

```bash
chmod +x /home/qtrobot/catkin_ws/src/qt_dss_app/src/QT-Robot-Dementia-Speech-System-Client-V3/launch.sh
```

Replace `/path/to/` with the actual path to the cloned repository (e.g. `/home/qtrobot/catkin_ws/src/qt_dss_app/src/`). 

#### Step 2: Create a desktop shortcut file

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

#### Step 3: Mark the desktop shortcut as trusted/executable

```bash
chmod +x ~/Desktop/qt-speech-system.desktop
```

On some desktop environments (such as LXDE, which QT robot uses), you may also need to right-click the file and select **"Trust this executable"** or **"Allow executing"** from the context menu.

#### Step 4: Test it

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

### 6. Pulling updates from the repository
- Navigate to the repository folder:
```bash
cd ~/catkin_ws/src/qt_dss_app/src/QT-Robot-Dementia-Speech-System-Client-V3/ 
```

- restore any git changes (this happens when using the executable launcherm which modifies the launcher script to add the venv activation):
```bash
git restore .
```

- Pull the latest changes:
```bash
git pull
```

- (Optional) Changing to a particular branch (if not using `main`):
```bash
git checkout branch_name
```

Replace `branch_name` with the actual branch name you want to switch to. Change branch_name to `main` if you want to switch back ßto the main branch.

- Make the launcher script executable again (since it was restored):
```bash
chmod +x /home/qtrobot/catkin_ws/src/qt_dss_app/src/QT-Robot-Dementia-Speech-System-Client-V3/launch.sh
```

- Mark the desktop shortcut as trusted/executable again (if needed):
```bash
chmod +x ~/Desktop/qt-speech-system.desktop
```

> [!TIP]
> After pulling updates, you can now launch the application using the desktop shortcut.
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

### Prepare the robot's voice

1. Click **Settings** in the top toolbar.
2. Select the microphone and click **Apply** if changing it.
3. Choose whether to save microphone audio using **Save conversation audio**.
4. Click **Try my voice** to hear the robot while adjusting Volume and Speed.
5. Click **Finish voice setup** when satisfied.
6. Click **Back to conversation** to return to the main screen.

Voice preview runs locally without starting a backend conversation, microphone capture, or recording. Speed and volume can be adjusted using the sliders, +/− buttons, or arrow keys.

Finishing voice setup or returning to the main screen lets the current sentence finish before stopping. Start Chat remains unavailable until the previous activity finishes.

To customize the story, edit `assets/voice_preview_story.txt`. Each nonempty line is spoken separately. Keep sentences short so adjustments and stopping remain responsive.

### Start a conversation

1. Click **Start Chat**. The robot greets the user and starts listening.
2. The user speaks; microphone audio streams to the backend.
3. Click the circular **Send** button or press **Enter** on the main screen when the user finishes speaking.
4. Wait while the robot thinks and responds. Send is unavailable during this time.
5. The robot automatically resumes listening after responding.
6. Open **Settings** if needed during the conversation, or use arrow keys to adjust speech.
7. Click **Stop Chat** to end the session manually. Listening stops, and outstanding speech finishes before cleanup completes.
8. When the backend completes the conversation, the robot finishes its final response and the application closes after a countdown.

### Keyboard shortcuts

| Key | Action | Available on |
|---|---|---|
| Enter / keypad Enter | Send when the Send button is enabled | Main screen |
| ↑ | Increase volume by 5 | Main and settings screens |
| ↓ | Decrease volume by 5 | Main and settings screens |
| → | Increase speech speed by 5 | Main and settings screens |
| ← | Decrease speech speed by 5 | Main and settings screens |

### Settings panel

| Setting | Description |
|---|---|
| **Microphone** | Select an input device and click **Apply**. The selection is used when the next session starts. |
| **Save conversation audio** | Save microphone input as a timestamped FLAC file in `recordings/`. |
| **Speed** | Adjust robot speech speed from 50–120. |
| **Volume** | Adjust speaker volume from 0–100. |
| **Try my voice** | Hear a repeating story while adjusting speed and volume before a conversation. |

All settings (except microphone) are applied live without needing to click any button. Settings are saved automatically and restored on the next launch.

## How the Backend Connection Works (For future reference)
```text
User starts speaking
  └─► _on_audio() / _pa_callback() fires every ~20ms
        ├─► _audio_buffer.extend(chunk)   [for "has_audio" check]
        └─► backend.send_audio_chunk()    [live stream to backend]
              └─► backend STT already transcribing in real-time
                    ├─► interim results → cancel pending task (no-op for qtrobot)
                    └─► final results  → stage_and_schedule()
                                           └─► audio_done=False → hold stt_staged

User clicks Send or presses Enter
  └─► pause_listening()  [stop accumulating]
  └─► reset_stt_staged_event()
  └─► send_audio_done()  → backend sets _audio_done=True
                              └─► staged_utterances non-empty → send stt_staged 
  └─► wait_for_stt_staged() unblocks immediately (STT already done!)
  └─► send_staged() → LLM responds with full transcript 

Robot sends "robot_send_staged"  ◄─── triggered after STT is staged                                
  └─► reply_now()
        ├─► flush_staged_utterances()  → commits user msg to DB
        └─► respond_to_user()
              ├─► consumer.response_method() → LLM call
              └─► consumer.send({"type": "llm_response", ...})  ← sent back to robot
```

---