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
      - Robot starts listening (STT accumulates transcript)
      - User clicks "Send" -> accumulated transcript sent to backend
      - Robot speaks the response (STT paused)
      - Robot finishes speaking -> back to step 1
    """

    def __init__(self, bus: EventBus, robot: RobotActions, stt: STTAccumulator, backend: BackendBridge):
        self._bus = bus
        self._robot = robot
        self._stt = stt
        self._backend = backend
        self._session_active = False

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
        Grabs accumulated STT transcript, sends to backend, robot speaks response.
        """
        if not self._session_active:
            self._bus.publish("error", "No active session.")
            return

        transcript = self._stt.get_and_clear_transcript()
        if not transcript:
            self._bus.publish("error", "Nothing to send. Please speak first.")
            return

        # Pause listening while we process
        self._stt.pause_listening()
        self._bus.publish("stt_final", "")  # Clear the transcript display

        # Publish user message to chat UI
        self._bus.publish("user_message", transcript)
        self._bus.publish("status", "Thinking...")

        # Run backend call + robot speech in background thread
        threading.Thread(target=self._process_turn, args=(transcript,), daemon=True).start()

    def _process_turn(self, transcript):
        """Background: send to backend, speak response, resume listening."""
        try:
            llm_start = time.perf_counter()

            response_text, response_emotion, current_scenario, next_scenario = (
                self._backend.send_transcript_and_wait(
                    transcript,
                    emotion=None,
                    timeout=settings.LLM_TIMEOUT,
                )
            )

            llm_time = time.perf_counter() - llm_start

            # Publish response to UI
            self._bus.publish(
                "llm_response",
                response_text,
                emotion=response_emotion,
                current_scenario=current_scenario,
                next_scenario=next_scenario,
                response_time=llm_time,
            )

            # For future use: calling the execute_actions function for any movement/emotion commands from the backend response
            # self._robot.execute_actions(response.get("actions"))

            # Robot speaks (blocking)
            self._bus.publish("status", "Speaking...")
            emotion = response_emotion.lower() if response_emotion else "neutral"
            self._robot.say(response_text, emotion)

        except Exception as e:
            self._bus.publish("error", f"Backend error: {e}")
            traceback.print_exc()

        finally:
            # Resume listening for next turn
            if self._session_active:
                self._stt.resume_listening()