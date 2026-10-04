import threading
from pathlib import Path


class VoicePreview:
    """Repeat a local story without starting a conversation or microphone."""

    INTRODUCTION = (
        "Let's set up my volume and talking speed first. "
        "You can adjust the sliders while I tell you a little story."
    )

    def __init__(self, robot, bus):
        self._robot = robot
        self._bus = bus
        self._story_path = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "voice_preview_story.txt"
        )
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def is_active(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def is_stopping(self):
        return self.is_active() and self._stop_event.is_set()

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set():
                self._bus.publish(
                    "voice_preview", "Finishing this sentence...", state="stopping"
                )
                self._stop_event.set()

    def _run(self):
        error = None
        try:
            lines = [
                line.strip()
                for line in self._story_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                raise ValueError("The voice preview story is empty.")
            if not self._robot.speech_available():
                raise RuntimeError("Connect to the QT robot to try its voice.")

            # Publish before stop() can publish its stopping message.
            with self._lock:
                if self._stop_event.is_set():
                    return
                self._bus.publish(
                    "voice_preview", "Adjust Volume and Speed as I speak.", state="playing"
                )

            if not self._stop_event.is_set():
                if not self._robot.say(self.INTRODUCTION, play_gestures=False):
                    raise RuntimeError("The robot could not speak the introduction.")
            # add a small delay between the introduction and the story
            if self._stop_event.wait(1.0):
                return
            
            while not self._stop_event.is_set():
                for line in lines:
                    if self._stop_event.is_set():
                        return
                    if not self._robot.say(line, play_gestures=False):
                        raise RuntimeError("The robot could not read the story.")
                    if self._stop_event.wait(0.4):
                        return

        except Exception as e:
            error = str(e)
        finally:
            with self._lock:
                self._bus.publish(
                    "voice_preview",
                    f"Voice preview error: {error}" if error else "Voice setup finished.",
                    state="error" if error else "idle",
                )
