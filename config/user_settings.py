"""
Manages persistent user settings stored in user_settings.json (gitignored).
Loaded once at startup; saved whenever settings change.

On load, the saved microphone device index is verified against the current
PyAudio device list by name. If the index has shifted (common with USB devices),
it is corrected automatically.
"""
import json
import os
import pyaudio
from typing import Optional


# Stored at the project root — outside version control (see .gitignore)
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "user_settings.json")
_SETTINGS_PATH = os.path.normpath(_SETTINGS_PATH)

# Keys that are allowed to be persisted (avoids accidentally saving credentials etc.)
_PERSISTED_KEYS = [
    "SPEECH_SPEED",
    "SPEECH_VOLUME",
    "TRANSCRIPT_FONT_SIZE",
    "MIC_SOURCE",
    "MIC_DEVICE_INDEX",
    "MIC_DEVICE_NAME",
    "RECORD_SESSION", 
]


def load_user_settings(settings_obj) -> None:
    """
    Load persisted values from user_settings.json into the given settings object.
    After loading, resolves the microphone index by name in case it has shifted
    since the settings were last saved. Silently does nothing if the file doesn't
    exist or is malformed.
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
        return

    # ── Resolve mic index by name ──
    # If MIC_SOURCE is "external" and we have a saved device name, scan the current
    # PyAudio device list to confirm the saved index still matches that name.
    # If the index has shifted, find the correct device by name and update the index.
    if getattr(settings_obj, "MIC_SOURCE", "default") == "external":
        saved_name = getattr(settings_obj, "MIC_DEVICE_NAME", None)
        saved_index = getattr(settings_obj, "MIC_DEVICE_INDEX", None)
        if saved_name:
            resolved_index = _resolve_mic_index_by_name(saved_name, saved_index)
            if resolved_index is not None:
                if resolved_index != saved_index:
                    print(
                        f"[user_settings] Mic index shifted: '{saved_name}' was at "
                        f"index {saved_index}, now at {resolved_index}. Updating."
                    )
                settings_obj.MIC_DEVICE_INDEX = resolved_index
            else:
                # Device not found at all — warn and fall back to built-in
                print(
                    f"[user_settings] Saved mic '{saved_name}' not found in current "
                    f"device list. Falling back to built-in mic."
                )
                settings_obj.MIC_SOURCE = "default"
                settings_obj.MIC_DEVICE_INDEX = None


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


def _resolve_mic_index_by_name(device_name: str, fallback_index) -> Optional[int]:
    """
    Scan current PyAudio input devices and return the index of the device
    whose name contains device_name (case-insensitive substring match).
    Returns fallback_index if no match is found but fallback_index itself is valid,
    or None if the device cannot be found at all.
    """
    try:
        pa = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0: # Only consider input devices
                    devices.append({"index": i, "name": info["name"]})
        finally:
            pa.terminate()

        # First: try exact index match to confirm it still has the right name
        if fallback_index is not None:
            try:
                fi = int(fallback_index)
                match = next((d for d in devices if d["index"] == fi), None)
                if match and device_name.lower() in match["name"].lower():
                    return fi  # Index is still correct — no change needed
            except (ValueError, TypeError):
                pass

        # Second: scan all devices for a name match (handles index shift)
        for d in devices:
            if device_name.lower() in d["name"].lower():
                return d["index"]

        return None  # Device not found anywhere

    except Exception as e:
        print(f"[user_settings] Could not scan audio devices: {e}")
        return fallback_index  # Best-effort: return what we have