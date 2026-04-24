import time
import threading
import traceback
import rospy

from services.event_bus import EventBus
from services.backend_client import BackendBridge
from services.stt_accumulator import STTAccumulator
from services.robot_actions import RobotActions
from config.settings import settings


class ChatController:
    """
    Orchestrates the turn-taking flow:
      - Robot starts listening (STT accumulates transcript and audio)
      - User clicks "Send" -> accumulated audio sent to backend
      - Robot speaks the response (STT paused)
      - Robot finishes speaking -> back to step 1
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
                self._stt.setup_ros_audio()
                self._stt.start_listening()
            except Exception as e:
                self._bus.publish("error", f"Failed to start: {e}")
                self._session_active = False
                traceback.print_exc()

        threading.Thread(target=_start, daemon=True).start()

    def stop_session(self):
        """Called when user clicks Stop Chat."""
        self._session_active = False
        self._stt.stop_listening()
        self._backend.stop()
        self._bus.publish("status", "Session ended.")

    # ------------------------------------------------------------------
    # Turn-taking: user sends accumulated transcript
    # ------------------------------------------------------------------

    def send_message(self):
        """
        Called when user clicks Send.
        Sends accumulated audio to backend, backend STT generates transcript and LLM response.
        Local STT transcript is used only for UI display.
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

        # Show local STT transcript in UI (display only, backend does the real STT)
        local_transcript = self._stt.get_and_clear_transcript()
        if local_transcript:
            self._bus.publish("user_message", local_transcript)

        self._bus.publish("stt_final", "")  # Clear transcript display
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
            self._backend.send_staged()

        except Exception as e:
            self._bus.publish("error", f"Failed to send audio: {e}")
            traceback.print_exc()
            if self._session_active:
                self._stt.resume_listening()

    def _on_llm_response_received(self, text, emotion, current_scenario, next_scenario):
        """
        Called from the asyncio loop thread when the backend sends an llm_response.
        Dispatches robot speech to a background thread.
        """
        if not self._session_active:
            return
        threading.Thread(
            target=self._process_response,
            args=(text, emotion, current_scenario, next_scenario),
            daemon=True
        ).start()

    def _process_response(self, response_text, response_emotion, current_scenario, next_scenario):
        """Background: publish response to UI, speak it, then resume listening."""
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
            self._robot.say(response_text, emotion)

        except Exception as e:
            self._bus.publish("error", f"Response error: {e}")
            traceback.print_exc()

        finally:
            if self._session_active:
                self._stt.resume_listening()