import threading

import pyaudio

#try:
import rospy
from audio_common_msgs.msg import AudioData
ROS_AVAILABLE = True
#except ImportError:
    #ROS_AVAILABLE = False

from config.settings import settings
from services.session_recorder import SessionRecorder


class ManualRecorder:
    """Capture the selected microphone without starting a chat."""

    def __init__(self, bus):
        self._bus = bus
        self._sample_rate = settings.AUDIO_RATE
        self._recorder = SessionRecorder(sample_rate=self._sample_rate)
        self._lock = threading.Lock()
        self._active = False
        self._stopping = False
        self._stop_event = threading.Event()
        self._audio_sub = None
        self._pa_stream = None
        self._pyaudio = None

    def is_active(self):
        with self._lock:
            return self._active

    def is_stopping(self):
        with self._lock:
            return self._stopping

    def start(self):
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._stopping = False
            self._stop_event.clear()
            self._bus.publish(
                "manual_recording", "Starting manual recording...", state="starting"
            )
            # Capture the applied microphone selection for this recording.
            threading.Thread(
                target=self._run,
                args=(settings.MIC_SOURCE, settings.MIC_DEVICE_INDEX),
                daemon=True,
            ).start()
        return True

    def stop(self):
        with self._lock:
            if not self._active or self._stopping:
                return
            self._stopping = True
            self._stop_event.set()
            self._bus.publish(
                "manual_recording", "Saving manual recording...", state="stopping"
            )

    def _open_microphone(self, mic_source, mic_device_index):
        if mic_source == "external":
            self._pyaudio = pyaudio.PyAudio()
            device_index = (
                int(mic_device_index) if mic_device_index is not None else None
            )
            self._pa_stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=self._pa_callback,
            )
            self._pa_stream.start_stream()
        else:
            self._audio_sub = rospy.Subscriber(
                '/qt_respeaker_app/channel0', AudioData, self._on_audio
            )

    def _pa_callback(self, in_data, frame_count, time_info, status):
        self._recorder.write(in_data)
        return (None, pyaudio.paContinue)

    def _on_audio(self, msg):
        self._recorder.write(bytes(msg.data))

    def _run(self, mic_source, mic_device_index):
        error = None
        output_path = None
        try:
            if not self._stop_event.is_set():
                if mic_source != "external" and not ROS_AVAILABLE:
                    raise RuntimeError(
                        "The robot microphone requires ROS. Select an external mic."
                    )
                self._recorder.start(manual=True)
                output_path = self._recorder.output_path
                self._open_microphone(mic_source, mic_device_index)
                with self._lock:
                    if not self._stop_event.is_set():
                        self._bus.publish(
                            "manual_recording", "Manual recording in progress.",
                            state="recording",
                        )
                self._stop_event.wait()
        except Exception as e:
            error = str(e)
        finally:
            # Stop capture before draining queued audio and closing the file.
            # Attempt every cleanup step even if an earlier one fails.
            cleanup_tasks = []
            if self._audio_sub is not None:
                cleanup_tasks.append(self._audio_sub.unregister)
            if self._pa_stream is not None:
                cleanup_tasks.extend((self._pa_stream.stop_stream, self._pa_stream.close))
            if self._pyaudio is not None:
                cleanup_tasks.append(self._pyaudio.terminate)
            cleanup_tasks.append(self._recorder.stop)

            for cleanup in cleanup_tasks:
                try:
                    cleanup()
                except Exception as e:
                    error = f"{error}; {e}" if error else str(e)

            self._audio_sub = None
            self._pa_stream = None
            self._pyaudio = None
            with self._lock:
                self._active = False
                self._stopping = False
                if error:
                    message = f"Manual recording failed: {error}"
                elif output_path:
                    message = f"Manual recording saved: {output_path}"
                else:
                    message = "Manual recording stopped."
                self._bus.publish(
                    "manual_recording", message,
                    state="error" if error else "idle",
                )