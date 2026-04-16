import queue
import threading
import time
import random
import rospy
from audio_common_msgs.msg import AudioData
from qt_robot_interface import srv

from services.audio_stream import MicrophoneStream
from services.event_bus import EventBus
from config.settings import settings


class STTAccumulator:
    """
    Continuous speech-to-text that accumulates transcript until explicitly sent.
    
    This accumulates all recognized text and publishes interim results to the
    EventBus. The user decides when to "send" the accumulated transcript.
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._aqueue = queue.Queue(maxsize=2000) 
        self._language = settings.DEFAULT_LANGUAGE
        self._audio_rate = settings.AUDIO_RATE
        self._model = settings.SPEECH_MODEL
        self._use_enhanced = settings.USE_ENHANCED_MODEL

        # Accumulated transcript from multiple final segments
        self._accumulated_text = ""
        self._lock = threading.Lock()

        # Control flags
        self._listening = False
        self._running = False
        self._listen_thread = None

        # ROS subscriber for audio
        self._audio_sub = None

        # Emotion service for listening feedback
        try:
            rospy.wait_for_service('/qt_robot/emotion/show', timeout=5)
            self._emotion_service = rospy.ServiceProxy('/qt_robot/emotion/show', srv.emotion_show)
        except Exception:
            self._emotion_service = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_ros_audio(self):
        """Subscribe to the robot's audio topic or open an external mic via PyAudio."""
        if settings.MIC_SOURCE == "external":
            self._setup_external_mic()
        else:
            # Default: subscribe to the ReSpeaker ROS topic
            self._audio_sub = rospy.Subscriber(
                '/qt_respeaker_app/channel0', AudioData, self._on_audio
            )

    def _setup_external_mic(self):
        """Open a PyAudio stream for the external USB microphone."""
        import pyaudio

        self._pyaudio = pyaudio.PyAudio()

        device_index = None
        if settings.MIC_DEVICE_INDEX is not None:
            device_index = int(settings.MIC_DEVICE_INDEX)

        # Log the device being used
        if device_index is not None:
            dev_info = self._pyaudio.get_device_info_by_index(device_index)
            rospy.loginfo(f"External mic: using device {device_index} — {dev_info['name']}")
        else:
            dev_info = self._pyaudio.get_default_input_device_info()
            rospy.loginfo(f"External mic: using system default — {dev_info['name']}")

        CHUNK = 1024  # frames per buffer

        self._pa_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._audio_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._pa_callback,
        )
        self._pa_stream.start_stream()

    def _pa_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — mirrors _on_audio but for external mic."""
        import pyaudio
        if self._listening:
            try:
                self._aqueue.put_nowait(in_data)
            except queue.Full:
                pass
        return (None, pyaudio.paContinue)

    def _on_audio(self, msg):
        """ROS audio callback — only queue data when listening."""
        if self._listening:
            try:
                self._aqueue.put_nowait(bytes(msg.data))
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Listening control
    # ------------------------------------------------------------------

    def start_listening(self):
        """Start continuous STT recognition."""
        if self._running:
            return

        self._running = True
        self._listening = True
        self._clear_accumulated()
        self._aqueue.queue.clear()

        self._listen_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._listen_thread.start()

        self._bus.publish("status", "Listening...")
        self._play_listening_emotion()

    def stop_listening(self):
        """Stop STT recognition (e.g., while robot is speaking)."""
        self._listening = False
        self._running = False
        # Cleanup PyAudio stream if using external mic
        if hasattr(self, '_pa_stream') and self._pa_stream is not None:
            self._pa_stream.stop_stream()
            self._pa_stream.close()
        if hasattr(self, '_pyaudio') and self._pyaudio is not None:
            self._pyaudio.terminate()
        # Put None to unblock the MicrophoneStream generator
        self._aqueue.put(None)

    def pause_listening(self):
        """Temporarily pause audio capture (robot speaking), keep thread alive."""
        self._listening = False

    def resume_listening(self):
        """Resume audio capture after robot finishes speaking."""
        self._clear_accumulated()
        self._aqueue.queue.clear()
        self._listening = True
        self._bus.publish("status", "Listening...")
        self._play_listening_emotion()

    # ------------------------------------------------------------------
    # Transcript access
    # ------------------------------------------------------------------

    def get_and_clear_transcript(self) -> str:
        """Called by the controller when user clicks Send."""
        with self._lock:
            text = self._accumulated_text.strip()
            self._accumulated_text = ""
        return text

    def _clear_accumulated(self):
        with self._lock:
            self._accumulated_text = ""

    # ------------------------------------------------------------------
    # Recognition loop
    # ------------------------------------------------------------------

    def _recognition_loop(self):
        """Runs in a background thread. Continuously recognizes and accumulates."""
        from google.cloud import speech

        while self._running:
            if not self._listening:
                time.sleep(0.1)
                continue

            try:
                # Clear stale audio
                while self._aqueue.qsize() > int(self._audio_rate / 512 / 2):
                    self._aqueue.get()

                client = speech.SpeechClient()
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self._audio_rate,
                    language_code=self._language,
                    model=self._model,
                    use_enhanced=self._use_enhanced,
                    enable_automatic_punctuation=True,
                )
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=True,
                    enable_voice_activity_events=True,
                )

                with MicrophoneStream(self._aqueue) as mic:
                    audio_gen = mic.generator()
                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_gen
                    )

                    # Google streaming STT has a ~5min limit per stream,
                    # so we use a timeout and re-open the stream
                    responses = client.streaming_recognize(
                        streaming_config, requests, timeout=settings.DEFAULT_TIMEOUT
                    )
                    self._process_responses(responses)

            except Exception as e:
                if self._running:
                    rospy.logwarn(f"STT stream error (will retry): {e}")
                    time.sleep(1.0)

    def _process_responses(self, responses):
        """Process streaming STT responses, accumulating final results."""
        for response in responses:
            if not self._running or not self._listening:
                break

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            if result.is_final:
                # Append to accumulated text
                with self._lock:
                    if self._accumulated_text:
                        self._accumulated_text += " " + transcript
                    else:
                        self._accumulated_text = transcript

                # Publish the full accumulated text so far
                with self._lock:
                    full_text = self._accumulated_text
                self._bus.publish("stt_final", full_text)
            else:
                # Publish interim for live UI feedback
                with self._lock:
                    prefix = self._accumulated_text
                if prefix:
                    display = prefix + " " + transcript
                else:
                    display = transcript
                self._bus.publish("stt_interim", display)

    # ------------------------------------------------------------------
    # Robot feedback
    # ------------------------------------------------------------------

    def _play_listening_emotion(self):
        """Show a listening emotion on the robot."""
        if self._emotion_service is None:
            return
        try:
            emotion_name = random.choice(settings.EMOTION_LISTENING)
            self._emotion_service(emotion_name)
        except Exception as e:
            rospy.logwarn(f"Listening emotion failed: {e}")