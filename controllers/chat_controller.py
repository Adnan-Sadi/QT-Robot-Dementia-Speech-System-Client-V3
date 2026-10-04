import threading
import traceback

from services.event_bus import EventBus
from services.backend_client import BackendBridge
from services.stt_accumulator import STTAccumulator
from services.session_recorder import SessionRecorder
from services.robot_actions import RobotActions
from services.voice_preview import VoicePreview
from config.settings import settings


class ChatController:
    """
    Orchestrates the turn-taking flow:
      - Robot starts listening (STT accumulates audio)
      - User clicks "Send" -> accumulated audio sent to backend
      - Robot speaks the response (STT paused)
      - Robot finishes speaking -> back to step 1 
    """

    def __init__(self, bus: EventBus, robot: RobotActions, stt: STTAccumulator, backend: BackendBridge):
        self._bus = bus
        self._robot = robot
        self._stt = stt
        self._backend = backend
        self._recorder = SessionRecorder(sample_rate=settings.AUDIO_RATE)
        self._stt.set_recorder(self._recorder)
        self._session_active = False
        self._pending_chat_ended = False  # set by _on_chat_ended, used by _process_response

        self._activity_lock = threading.RLock()
        self._chat_workers = 0
        self._chat_idle = threading.Event()
        self._chat_idle.set()
        self._session_stopping = False
        self._voice_preview = VoicePreview(robot, bus)

        # backend llm_responses are handled here
        self._backend.set_response_callback(self._on_llm_response_received)
        self._backend.set_chat_ended_callback(self._on_chat_ended) 

    # ------------------------------------------------------------------
    # Voice preview and background activity
    # ------------------------------------------------------------------

    def is_busy(self):
        with self._activity_lock:
            return (
                self._session_active
                or self._session_stopping
                or self._chat_workers > 0
                or self._voice_preview.is_active()
            )

    def is_session_stopping(self):
        with self._activity_lock:
            return self._session_stopping

    def is_voice_preview_active(self):
        return self._voice_preview.is_active()

    def is_voice_preview_stopping(self):
        return self._voice_preview.is_stopping()

    def start_voice_preview(self):
        with self._activity_lock:
            if self.is_busy():
                self._bus.publish("error", "Please finish the current activity first.")
                return False
            return self._voice_preview.start()

    def stop_voice_preview(self):
        self._voice_preview.stop()

    def _start_chat_worker(self, target, args=()):
        """Track workers so preview cannot overlap unfinished chat work."""
        with self._activity_lock:
            if not self._session_active:
                return False
            self._chat_workers += 1
            self._chat_idle.clear()

        def _run():
            try:
                if self._session_active:
                    target(*args)
            finally:
                with self._activity_lock:
                    self._chat_workers -= 1
                    if self._chat_workers == 0:
                        self._chat_idle.set()

        threading.Thread(target=_run, daemon=True).start()
        return True

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def is_session_active(self) -> bool:
        return self._session_active

    def start_session(self):
        """Called when user clicks Start Chat."""
        with self._activity_lock:
            if self.is_busy():
                self._bus.publish("error", "Please finish the current activity first.")
                return False
            self._session_active = True
            self._pending_chat_ended = False

        self._bus.publish("status", "Connecting to backend...")

        def _start():
            try:
                self._backend.start()
                if not self._session_active:
                    return
                self._bus.publish("status", "Connected. Starting listener...")

                # Start recording session audio to WAV file (if enabled)
                if settings.RECORD_SESSION:
                    self._recorder.start()

                self._stt.setup_ros_audio()
                if not self._session_active:
                    return

                # Play wakeup gesture and speak greeting before listening starts.
                self._robot.greet(settings.GREETING_TEXT)
                if not self._session_active:
                    return

                # Publish greeting to transcript panel
                self._bus.publish("llm_response", settings.GREETING_TEXT, emotion="happy", current_scenario=None, next_scenario=None)

                with self._activity_lock:
                    if self._session_active:
                        self._stt.start_listening()
                        self._bus.publish("status", "Listening...")

            except Exception as e:
                self._bus.publish("error", f"Failed to start: {e}")
                traceback.print_exc()
                self.stop_session()

        self._start_chat_worker(_start)
        return True

    def stop_session(self):
        """Stop listening now; clean up after outstanding chat workers finish."""
        with self._activity_lock:
            if self._session_stopping:
                return
            self._session_active = False
            self._pending_chat_ended = False
            self._session_stopping = True

        try:
            self._stt.stop_listening()
        except Exception as e:
            self._bus.publish("error", f"Failed to stop listening: {e}")
        self._bus.publish("status", "Finishing conversation...")

        def _stop():
            try:
                self._chat_idle.wait()
                self._stt.stop_listening()
                self._stt._clear_audio_buffer()  # Discard any stale audio so next session starts clean

                # Stop session recording if active
                if self._recorder is not None:
                    self._recorder.stop()

                self._backend.stop()             # stop() now resets the backend for a future start()
                self._bus.publish("status", "Session ended.")
            except Exception as e:
                self._bus.publish("error", f"Failed to stop: {e}")
            finally:
                with self._activity_lock:
                    self._session_stopping = False

        threading.Thread(target=_stop, daemon=True).start()

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

    # ------------------------------------------------------------------
    # Turn-taking: user sends accumulated audio
    # ------------------------------------------------------------------

    def send_message(self):
        """Called when user clicks Send."""
        if not self._session_active:
            self._bus.publish("error", "No active session.")
            return False

        # Check that there's something in the buffer (user did speak)
        if not self._stt.has_audio():
            self._bus.publish("error", "Nothing to send. Please speak first.")
            return False

        # Pause listening (stop accumulating + stop live streaming)
        self._stt.pause_listening()

        self._bus.publish("status", "Thinking...")
        return self._start_chat_worker(self._dispatch_audio)

    def _dispatch_audio(self, audio_data=None):  # audio_data no longer needed
        """Background: signal backend that audio is done, wait for STT, then trigger LLM."""
        try:
            # Reset event before signalling done (in case a stale stt_staged exists)
            self._backend.reset_stt_staged_event()

            # Audio was already streamed live — just tell backend recording is complete
            self._backend.send_audio_done()

            # Wait for backend to confirm all STT results are staged
            staged_ok = self._backend.wait_for_stt_staged(timeout=5.0) # wait 5 seconds, should be enough since I am streaming the audio now
            if not staged_ok:
                # No STT transcript was produced — the user was silent (or only background noise).
                # Abort this turn silently and go back to listening.
                print("[ChatController] No STT results staged — user was silent. Aborting turn.")
                with self._activity_lock:
                    if self._session_active:
                        self._stt.resume_listening()
                        self._bus.publish("status", "Listening...")
                return

            if self._session_active:
                self._backend.send_staged()

        except Exception as e:
            self._bus.publish("error", f"Failed to send audio: {e}")
            traceback.print_exc()
            with self._activity_lock:
                if self._session_active:
                    self._stt.resume_listening()
                    self._bus.publish("status", "Listening...")

    def _on_llm_response_received(self, text, emotion, current_scenario, next_scenario):
        """
        Called from the asyncio loop thread when the backend sends an llm_response.
        Dispatches robot speech to a background thread.
        """
        print(f"[ChatController] llm_response received: text='{text[:50]}', emotion={emotion}, scenario={current_scenario}")
        if not self._session_active:
            return
        self._start_chat_worker(
            self._process_response,
            args=(text, emotion, current_scenario, next_scenario),
        )

    def _process_response(self, response_text, response_emotion, current_scenario, next_scenario):
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
            self._robot.say(response_text, emotion)  # blocks until speech is fully done

        except Exception as e:
            self._bus.publish("error", f"Response error: {e}")
            traceback.print_exc()

        finally:
            if self._pending_chat_ended:
                # Robot has finished speaking the final response — now it is safe to shut down
                self._pending_chat_ended = False
                self.stop_session()
                self._bus.publish("chat_ended", "")
            else:
                with self._activity_lock:
                    if self._session_active:
                        self._stt.resume_listening()
                        self._bus.publish("status", "Listening...") 

    # ------------------------------------------------------------------
    # Session Closure: backend signals chat_ended
    # ------------------------------------------------------------------
    def _on_chat_ended(self):
        """
        Called from the asyncio loop thread when the backend sends a chat_ended signal.
        We do NOT shut down immediately here — the robot may still be speaking the final
        response. Instead, we set a flag so _process_response can trigger shutdown after
        robot.say() returns.
        """
        print("[ChatController] chat_ended received — will close after robot finishes speaking.")
        with self._activity_lock:
            if self._session_active:
                self._pending_chat_ended = True

        # Safety fallback: if _process_response never picks this up
        # shut down after a timeout.
        # I only need this when I am not running the client on the robot itself, because the robot's say() function is blocking the thread until the robot finishes speaking.
        # def _fallback_shutdown():
        #     import time
        #     time.sleep(25)  # wait up to 25s for robot to finish speaking
        #     if self._pending_chat_ended:
        #         print("[ChatController] chat_ended fallback shutdown triggered.")
        #         self._pending_chat_ended = False
        #         self.stop_session()
        #         self._bus.publish("chat_ended", "")
        
        # threading.Thread(target=_fallback_shutdown, daemon=True).start()