"""
SessionRecorder — lightweight WAV recording of the user's microphone input.

Receives PCM audio captured by STTAccumulator and writes it to a WAV file
in the /recordings folder.

Usage:
    recorder = SessionRecorder(sample_rate=16000)
    recorder.start()
    recorder.write(audio_chunk)
    recorder.stop()
"""

import datetime
import os
import queue
import threading
import wave


# Recordings are stored here — folder is created automatically, gitignored
_RECORDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "recordings")
)

_STOP_WRITER = object()


class SessionRecorder:
    """
    Writes microphone audio chunks to a timestamped WAV file.

    Audio is written on a background thread so the PyAudio or ROS audio
    callback is never blocked by disk I/O.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        sample_rate: Rate of the PCM chunks passed to write().
        """
        self._sample_rate = sample_rate

        self._wav_file = None
        self._thread = None
        self._queue = None
        self._state_lock = threading.Lock()
        self._accepting_audio = False
        self._output_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open a WAV file and start the background writer."""
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                print("[SessionRecorder] Already recording — ignoring start().")
                return

            os.makedirs(_RECORDINGS_DIR, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_path = os.path.join(
                _RECORDINGS_DIR,
                f"session_{timestamp}.wav",
            )

            wav_file = wave.open(self._output_path, "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes per sample
            wav_file.setframerate(self._sample_rate)

            audio_queue = queue.SimpleQueue()

            self._wav_file = wav_file
            self._queue = audio_queue
            self._accepting_audio = True
            self._thread = threading.Thread(
                target=self._record_loop,
                args=(wav_file, audio_queue),
                daemon=True,
            )
            self._thread.start()

        print(f"[SessionRecorder] Recording started → {self._output_path}")

    def write(self, pcm_bytes: bytes):
        """
        Queue one raw mono int16 PCM chunk for recording.

        This method is safe to call from a PyAudio or ROS callback.
        It does nothing when recording is not active.
        """
        if not pcm_bytes:
            return

        with self._state_lock:
            if not self._accepting_audio or self._queue is None:
                return
            self._queue.put(pcm_bytes)

    def stop(self):
        """Stop accepting audio and finalise the WAV file."""
        with self._state_lock:
            if not self._accepting_audio or self._thread is None:
                return

            self._accepting_audio = False
            thread = self._thread
            self._queue.put(_STOP_WRITER)

        # Wait until every queued chunk has been written and the WAV header
        # has been finalised.
        thread.join()

        with self._state_lock:
            self._thread = None
            self._queue = None
            self._wav_file = None

        print(f"[SessionRecorder] Recording saved → {self._output_path}")

    @property
    def output_path(self):
        """Path of the last or current recording, or None if never started."""
        return self._output_path

    # ------------------------------------------------------------------
    # Background recording loop
    # ------------------------------------------------------------------

    @staticmethod
    def _record_loop(wav_file, audio_queue):
        """Write queued PCM chunks without blocking the capture callback."""
        try:
            while True:
                data = audio_queue.get()
                if data is _STOP_WRITER:
                    break

                # wave.close() finalises the header at the end.
                wav_file.writeframesraw(data)
        finally:
            wav_file.close()