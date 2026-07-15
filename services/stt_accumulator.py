import queue
import threading
import random
# making rospy optional for testing without ROS
#try:
import rospy
from audio_common_msgs.msg import AudioData
from qt_robot_interface import srv
ROS_AVAILABLE = True
# except ImportError:
#     ROS_AVAILABLE = False

from services.event_bus import EventBus
from config.settings import settings

import numpy as np
import librosa


class STTAccumulator:
    """
    Captures raw PCM audio from the microphone (ROS topic or external USB mic)
    and accumulates it in a buffer. When the user clicks Send, the buffer is
    retrieved, resampled to 16 kHz, and forwarded to the backend for STT and
    LLM response generation.
    """

    def __init__(self, bus: EventBus, backend=None):
        self._bus = bus
        self._audio_rate = settings.AUDIO_RATE

        self._lock = threading.Lock()

        # Control flags
        self._listening = False
        self._running = False

        # ROS subscriber for audio
        self._audio_sub = None

        self._backend = backend          # BackendBridge — for sending audio on Send click
        self._audio_buffer = bytearray() # Raw PCM accumulation for backend audio sending

        # Emotion service for listening feedback
        self._emotion_service = None
        if ROS_AVAILABLE:
            try:
                rospy.wait_for_service('/qt_robot/emotion/show', timeout=5)
                self._emotion_service = rospy.ServiceProxy('/qt_robot/emotion/show', srv.emotion_show)
            except Exception:
                self._emotion_service = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_ros_audio(self):
        """Subscribe to the robot's audio topic."""
        if settings.MIC_SOURCE == "external":
            self._setup_external_mic()
        else:
            if ROS_AVAILABLE:
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
            print(f"External mic: using device {device_index} — {dev_info['name']}")
        else:
            dev_info = self._pyaudio.get_default_input_device_info()
            print(f"External mic: using system default — {dev_info['name']}")

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
        """PyAudio callback — accumulates audio from the external mic."""
        import pyaudio
        if self._listening:
            # Accumulate for backend sending on Send click
            with self._lock:
                self._audio_buffer.extend(in_data)
        return (None, pyaudio.paContinue)

    def _on_audio(self, msg):
        """ROS audio callback — only accumulate data when listening."""
        if self._listening:
            chunk = bytes(msg.data)
            # Accumulate for backend sending on Send click
            with self._lock:
                self._audio_buffer.extend(chunk)

    # ------------------------------------------------------------------
    # Listening control
    # ------------------------------------------------------------------

    def start_listening(self):
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        self._listening = True
        self._clear_audio_buffer()

        self._bus.publish("status", "Listening...")
        self._play_listening_emotion()

    def stop_listening(self):
        """Stop audio capture entirely (e.g., session ended)."""
        self._listening = False
        self._running = False
        # Cleanup PyAudio stream if using external mic
        if hasattr(self, '_pa_stream') and self._pa_stream is not None:
            self._pa_stream.stop_stream()
            self._pa_stream.close()
            self._pa_stream = None
        if hasattr(self, '_pyaudio') and self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def pause_listening(self):
        """Temporarily pause audio capture (robot is speaking), keep state alive."""
        self._listening = False

    def resume_listening(self):
        """Resume audio capture after robot finishes speaking."""
        self._clear_audio_buffer()
        self._listening = True
        self._bus.publish("status", "Listening...")
        self._play_listening_emotion()

    # ------------------------------------------------------------------
    # Audio buffer access
    # ------------------------------------------------------------------

    def get_and_clear_audio_buffer(self) -> bytes:
        """Called when user clicks Send — returns all accumulated PCM resampled to 16000 Hz."""
        with self._lock:
            data = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()
        return self._resample_to_16k(data)

    def _clear_audio_buffer(self):
        with self._lock:
            self._audio_buffer = bytearray()

    def _resample_to_16k(self, pcm_bytes: bytes) -> bytes:
        """Resample raw PCM int16 bytes from self._audio_rate to 16000 Hz.
        
        Uses librosa.resample with float32 normalisation to avoid int16 clipping distortion.
        """
        if self._audio_rate == 16000:
            return pcm_bytes

        # Convert int16 PCM → float32 normalised to [-1.0, 1.0]
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample using librosa 
        audio_resampled = librosa.resample(audio, orig_sr=self._audio_rate, target_sr=16000)

        # Clip to prevent any overflow, then convert back to int16
        audio_resampled = np.clip(audio_resampled, -1.0, 1.0)
        return (audio_resampled * 32767).astype(np.int16).tobytes()

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
            print(f"Listening emotion failed: {e}")

    # ------------------------------------------------------------------
    # List audio input devices
    # ------------------------------------------------------------------
    @staticmethod
    def list_audio_input_devices() -> list:
        """
        Returns a list of dicts describing available PyAudio input devices.
        Each dict has: {'index': int, 'name': str, 'sample_rate': int}
        Safe to call before ROS is initialised.
        """
        import pyaudio
        devices = []
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'sample_rate': int(info['defaultSampleRate']),
                    })
        finally:
            pa.terminate()
        return devices
