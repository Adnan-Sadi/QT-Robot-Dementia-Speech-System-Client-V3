import time
import threading
import traceback

from services.event_bus import EventBus
from services.backend_client import BackendBridge
from services.stt_accumulator import STTAccumulator
from services.robot_actions import RobotActions
from config.settings import settings
from config.user_settings import save_user_settings


class ChatController:
    """
    Orchestrates the turn-taking flow:
      - Robot starts listening (STT accumulates audio)
      - User clicks "Send" -> accumulated audio sent to backend
      - Robot speaks the response (STT paused)
      - Robot finishes speaking -> back to step 1 (unless close_session is True)
    """

    def __init__(self, bus: EventBus, robot: RobotActions, stt: STTAccumulator, backend: BackendBridge):
        self._bus = bus
        self._robot = robot
        self._stt = stt
        self._backend = backend
        self._session_active = False

        # backend llm_responses are handled here
        self._backend.set_response_callback(self._on_llm_response_received)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def is_session_active(self) -> bool:
        return self._session_active

    def start_session(self):
        """Called when user clicks Start Chat."""
        if self._session_active:
            self._bus.publish("error", "Session already active.")
            return

        self._session_active = True
        self._bus.publish("status", "Connecting to backend...")

        def _start():
            try:
                self._backend.start()
                self._bus.publish("status", "Connected. Starting listener...")

                # Play wakeup gesture and speak greeting before listening starts.
                self._robot.greet(settings.GREETING_TEXT)

                self._stt.setup_ros_audio()
                self._stt.start_listening()
                
            except Exception as e:
                self._bus.publish("error", f"Failed to start: {e}")
                self._session_active = False
                traceback.print_exc()

        threading.Thread(target=_start, daemon=True).start()

    def stop_session(self):
        """Called when user clicks Stop Chat (or triggered automatically on close_session)."""
        self._session_active = False
        self._stt.stop_listening()
        self._stt._clear_audio_buffer()  # Discard any stale audio so next session starts clean
        self._backend.stop()             # stop() now resets the backend for a future start()
        self._bus.publish("status", "Session ended.")

    # ------------------------------------------------------------------
    # Runtime settings
    # ------------------------------------------------------------------
    def apply_settings(self, mic_device_index, mic_source, speech_speed, volume):
        """
        Called from the Settings panel to apply runtime configuration.
        Any argument can be None — only non-None values are applied.
        mic_device_index: int or None (None = system default)
        mic_source: "default" (ReSpeaker ROS topic) or "external" (PyAudio), or None to skip
        speech_speed: int (e.g. 50–200), or None to skip
        volume: int (0–100), or None to skip
        """
        # Only update mic settings if mic_source is explicitly provided
        if mic_source is not None:
            settings.MIC_SOURCE = mic_source
            settings.MIC_DEVICE_INDEX = mic_device_index

        # Apply speech settings immediately (safe to call any time)
        if speech_speed is not None:
            self._robot.configure_speech_speed(speech_speed)

        if volume is not None:
            self._robot.configure_volume(volume)

        # Persist the updated settings to disk
        save_user_settings(settings)

    # ------------------------------------------------------------------
    # Turn-taking: user sends accumulated audio
    # ------------------------------------------------------------------

    def send_message(self):
        """
        Called when user clicks Send.
        Sends accumulated audio to backend; backend STT generates transcript and LLM response.
        """
        if not self._session_active:
            self._bus.publish("error", "No active session.")
            return

        audio_data = self._stt.get_and_clear_audio_buffer()
        if not audio_data:
            self._bus.publish("error", "Nothing to send. Please speak first.")
            return

        # Pause listening while we process
        self._stt.pause_listening()

        self._bus.publish("status", "Thinking...")

        threading.Thread(target=self._dispatch_audio, args=(audio_data,), daemon=True).start()

    def _dispatch_audio(self, audio_data: bytes):
        """Background: send accumulated audio to backend, then trigger LLM response."""
        try:
            # Send audio in chunks (backend STT accumulates them)
            CHUNK = 4096
            for i in range(0, len(audio_data), CHUNK):
                self._backend.send_audio_chunk(audio_data[i:i + CHUNK], sample_rate=16000)

            # Tell the backend to now flush staged utterances and generate the LLM response
            time.sleep(2.0) # adding a small delay for backend to flush, I need to improve this behavior
            self._backend.send_staged()

        except Exception as e:
            self._bus.publish("error", f"Failed to send audio: {e}")
            traceback.print_exc()
            if self._session_active:
                self._stt.resume_listening()

    def _on_llm_response_received(self, text, emotion, current_scenario, next_scenario, close_session=False):
        """
        Called from the asyncio loop thread when the backend sends an llm_response.
        Dispatches robot speech to a background thread.
        """
        print(f"[ChatController] llm_response received: text='{text[:50]}', emotion={emotion}, scenario={current_scenario}, close_session={close_session}")
        if not self._session_active:
            return
        threading.Thread(
            target=self._process_response,
            args=(text, emotion, current_scenario, next_scenario, close_session),
            daemon=True
        ).start()

    def _process_response(self, response_text, response_emotion, current_scenario, next_scenario, close_session):
        """Background: publish response to UI, speak it, then resume listening or close."""
        try:
            self._bus.publish(
                "llm_response",
                response_text,
                emotion=response_emotion,
                current_scenario=current_scenario,
                next_scenario=next_scenario,
            )
            self._bus.publish("status", "Speaking...")
            emotion = response_emotion.lower() if response_emotion else "neutral"

            # This blocks until the robot finishes speaking — close_session check comes after
            self._robot.say(response_text, emotion)

        except Exception as e:
            self._bus.publish("error", f"Response error: {e}")
            traceback.print_exc()

        finally:
            if close_session:
                # Robot has finished speaking — now trigger the shutdown sequence
                self.stop_session()
                self._bus.publish("close_session", "")
            elif self._session_active:
                self._stt.resume_listening()