"""
SessionRecorder — lightweight WAV recording of the user's microphone input.

Opens a separate PyAudio input stream (independent of the STT accumulator)
and writes raw PCM frames to a WAV file in the /recordings folder.

Usage:
    recorder = SessionRecorder(device_index=None, sample_rate=16000)
    recorder.start()
    # ... session runs ...
    recorder.stop()   # finalises the WAV file

The output folder is created automatically on first use.
"""

import os
import wave
import threading
import datetime

import pyaudio


# Recordings are stored here — folder is created automatically, gitignored
_RECORDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "recordings")
)

_CHUNK = 1024  # frames per buffer


class SessionRecorder:
    """
    Records microphone audio to a timestamped WAV file inside the /recordings folder.
    Uses a background thread to prevent blocking the main/UI thread.
    """

    def __init__(self, device_index=None, sample_rate: int = 16000):
        """
        device_index: PyAudio device index to record from (None = system default).
        sample_rate:  Capture rate in Hz. Should match the mic's native rate to
                      avoid any on-the-fly resampling overhead.
        """
        self._device_index = device_index
        self._sample_rate = sample_rate

        self._pa = None
        self._stream = None
        self._wav_file = None
        self._thread = None
        self._stop_event = threading.Event()
        self._output_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open the audio stream and start writing to a new WAV file."""
        if self._thread and self._thread.is_alive():
            print("[SessionRecorder] Already recording — ignoring start().")
            return

        # Ensure the recordings folder exists
        os.makedirs(_RECORDINGS_DIR, exist_ok=True)

        # Build a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = os.path.join(_RECORDINGS_DIR, f"session_{timestamp}.wav")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print(f"[SessionRecorder] Recording started → {self._output_path}")

    def stop(self):
        """Signal the recording thread to stop and wait for it to finalise the file."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        print(f"[SessionRecorder] Recording saved → {self._output_path}")

    @property
    def output_path(self):
        """Path of the last (or current) recording file, or None if not yet started."""
        return self._output_path

    # ------------------------------------------------------------------
    # Background recording loop
    # ------------------------------------------------------------------

    def _record_loop(self):
        """Runs in a daemon thread. Opens PyAudio, writes frames, closes on stop."""
        try:
            self._pa = pyaudio.PyAudio()

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=_CHUNK,
            )

            with wave.open(self._output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self._sample_rate)

                while not self._stop_event.is_set():
                    try:
                        data = self._stream.read(_CHUNK, exception_on_overflow=False)
                        wf.writeframes(data)
                    except OSError as e:
                        print(f"[SessionRecorder] Read error (ignored): {e}")

        except Exception as e:
            print(f"[SessionRecorder] Failed to start recording: {e}")
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None