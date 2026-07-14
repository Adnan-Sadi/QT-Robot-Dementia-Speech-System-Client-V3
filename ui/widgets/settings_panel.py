import customtkinter as ctk
import pyaudio

from config.settings import settings
from services.stt_accumulator import STTAccumulator


class SettingsPanel(ctk.CTkFrame):
    """
    An always-visible settings panel displayed on the left side of the main window.
    Allows adjusting microphone, speech speed, volume, and text size.
    """

    # Label shown in the dropdown for the robot's built-in ReSpeaker mic
    _ROS_MIC_LABEL = "QT Robot built-in mic (ReSpeaker)"

    def __init__(self, master, controller):
        super().__init__(master, fg_color=("gray90", "gray17"), corner_radius=8)
        self._controller = controller

        # Enumerate PyAudio devices once on open
        self._devices = STTAccumulator.list_audio_input_devices()

        # Build the dropdown values: built-in first, then detected PyAudio devices
        self._dropdown_values = [self._ROS_MIC_LABEL] + [
            f"[{d['index']}] {d['name']} ({d['sample_rate']} Hz)"
            for d in self._devices
        ]

        # Determine current selection to pre-fill the dropdown
        current_label = self._current_mic_label()

        # ── Layout ──
        self.grid_columnconfigure(0, weight=1)

        # Panel title
        ctk.CTkLabel(self, text="⚙  Settings", font=("", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 8)
        )

        # Section: Microphone
        ctk.CTkLabel(self, text="Microphone", font=("", 12, "bold")).grid(
            row=1, column=0, sticky="w", padx=16, pady=(8, 2)
        )
        ctk.CTkLabel(self, text="Input device:").grid(
            row=2, column=0, sticky="w", padx=16, pady=2
        )
        self._mic_var = ctk.StringVar(value=current_label)
        self._mic_dropdown = ctk.CTkOptionMenu(
            self,
            values=self._dropdown_values,
            variable=self._mic_var,
            width=220,
        )
        self._mic_dropdown.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))

        # Section: Speech
        ctk.CTkLabel(self, text="Robot Speech", font=("", 12, "bold")).grid(
            row=4, column=0, sticky="w", padx=16, pady=(8, 2)
        )

        # Speed slider
        ctk.CTkLabel(self, text="Speed:").grid(row=5, column=0, sticky="w", padx=16, pady=2)
        speed_frame = ctk.CTkFrame(self, fg_color="transparent")
        speed_frame.grid(row=6, column=0, sticky="ew", padx=16, pady=(0, 8))
        speed_frame.grid_columnconfigure(0, weight=1)

        self._speed_var = ctk.IntVar(value=settings.SPEECH_SPEED)
        self._speed_slider = ctk.CTkSlider(
            speed_frame, from_=50, to=200, number_of_steps=150,
            variable=self._speed_var,
            command=lambda v: self._speed_label.configure(text=str(int(v)))
        )
        self._speed_slider.grid(row=0, column=0, sticky="ew")
        self._speed_label = ctk.CTkLabel(speed_frame, text=str(settings.SPEECH_SPEED), width=36)
        self._speed_label.grid(row=0, column=1, padx=(8, 0))

        # Volume slider
        ctk.CTkLabel(self, text="Volume:").grid(row=7, column=0, sticky="w", padx=16, pady=2)
        vol_frame = ctk.CTkFrame(self, fg_color="transparent")
        vol_frame.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 8))
        vol_frame.grid_columnconfigure(0, weight=1)

        # Default volume from settings (fallback 80)
        self._vol_var = ctk.IntVar(value=getattr(settings, 'SPEECH_VOLUME', 80))
        self._vol_slider = ctk.CTkSlider(
            vol_frame, from_=0, to=100, number_of_steps=100,
            variable=self._vol_var,
            command=lambda v: self._vol_label.configure(text=str(int(v)))
        )
        self._vol_slider.grid(row=0, column=0, sticky="ew")
        self._vol_label = ctk.CTkLabel(vol_frame, text=str(self._vol_var.get()), width=36)
        self._vol_label.grid(row=0, column=1, padx=(8, 0))

        # Section: Text Size
        ctk.CTkLabel(self, text="Text Size", font=("", 12, "bold")).grid(
            row=9, column=0, sticky="w", padx=16, pady=(8, 2)
        )
        ctk.CTkLabel(self, text="Font size:").grid(row=10, column=0, sticky="w", padx=16, pady=2)
        size_frame = ctk.CTkFrame(self, fg_color="transparent")
        size_frame.grid(row=11, column=0, sticky="ew", padx=16, pady=(0, 8))
        size_frame.grid_columnconfigure(0, weight=1)

        self._font_size_var = ctk.IntVar(value=getattr(settings, 'TRANSCRIPT_FONT_SIZE', 13))
        self._font_size_slider = ctk.CTkSlider(
            size_frame, from_=10, to=28, number_of_steps=18,
            variable=self._font_size_var,
            command=self._on_font_size_change
        )
        self._font_size_slider.grid(row=0, column=0, sticky="ew")
        self._font_size_label = ctk.CTkLabel(size_frame, text=str(self._font_size_var.get()), width=36)
        self._font_size_label.grid(row=0, column=1, padx=(8, 0))

        # ── Apply button ──
        ctk.CTkButton(self, text="Apply Settings", command=self._on_apply).grid(
            row=12, column=0, padx=16, pady=(16, 16), sticky="ew"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_mic_label(self) -> str:
        """Return the dropdown label that matches the current settings."""
        if settings.MIC_SOURCE != "external":
            return self._ROS_MIC_LABEL
        # Try to match by device index
        current_index = settings.MIC_DEVICE_INDEX
        if current_index is not None:
            current_index = int(current_index)
            for d in self._devices:
                if d['index'] == current_index:
                    return f"[{d['index']}] {d['name']} ({d['sample_rate']} Hz)"
        # Fallback: no match found, show built-in
        return self._ROS_MIC_LABEL

    def _on_font_size_change(self, value):
        """Called live as the font size slider moves — updates label and notifies master."""
        size = int(value)
        self._font_size_label.configure(text=str(size))
        # Notify the main window to update the transcript font
        self.master.set_transcript_font_size(size)

    def _on_apply(self):
        """Read all controls and push values to the controller."""
        selected = self._mic_var.get()

        if selected == self._ROS_MIC_LABEL:
            mic_source = "default"
            mic_device_index = None
        else:
            # Parse the index out of "[N] Name (Hz)"
            mic_source = "external"
            try:
                mic_device_index = int(selected.split("]")[0].lstrip("["))
            except (ValueError, IndexError):
                mic_device_index = None

        speech_speed = int(self._speed_var.get())
        volume = int(self._vol_var.get())
        font_size = int(self._font_size_var.get())

        self._controller.apply_settings(
            mic_device_index=mic_device_index,
            mic_source=mic_source,
            speech_speed=speech_speed,
            volume=volume,
        )

        # Update in-memory defaults so sliders are correct if panel is re-opened
        settings.SPEECH_SPEED = speech_speed
        settings.SPEECH_VOLUME = volume
        settings.TRANSCRIPT_FONT_SIZE = font_size