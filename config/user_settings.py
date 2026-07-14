"""
Manages persistent user settings stored in user_settings.json (gitignored).
Loaded once at startup; saved whenever settings change.
"""
import json
import os

# Stored next to this file — outside version control (see .gitignore)
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "user_settings.json")
_SETTINGS_PATH = os.path.normpath(_SETTINGS_PATH)

# Keys that are allowed to be persisted (avoids accidentally saving credentials etc.)
_PERSISTED_KEYS = [
    "SPEECH_SPEED",
    "SPEECH_VOLUME",
    "TRANSCRIPT_FONT_SIZE",
    "MIC_SOURCE",
    "MIC_DEVICE_INDEX",
]


def load_user_settings(settings_obj) -> None:
    """
    Load persisted values from user_settings.json into the given settings object.
    Silently does nothing if the file doesn't exist or is malformed.
    """
    if not os.path.exists(_SETTINGS_PATH):
        return
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in _PERSISTED_KEYS:
            if key in data and data[key] is not None:
                setattr(settings_obj, key, data[key])
    except Exception as e:
        print(f"[user_settings] Could not load user settings: {e}")


def save_user_settings(settings_obj) -> None:
    """
    Save the current values of persisted keys to user_settings.json.
    """
    try:
        data = {key: getattr(settings_obj, key, None) for key in _PERSISTED_KEYS}
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[user_settings] Could not save user settings: {e}")