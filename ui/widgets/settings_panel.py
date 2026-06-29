import customtkinter as ctk
import pyaudio

from config.settings import settings
from services.stt_accumulator import STTAccumulator


class SettingsPanel(ctk.CTkToplevel):
    """
    A pop-up settings window for adjusting microphone, speech speed, and volume.
    Opens from the main toolbar. Changes are applied immediately via the controller.
    """

    # Label shown in the dropdown for the robot's built-in ReSpeaker mic
    _ROS_MIC_LABEL = "QT Robot built-in mic (ReSpeaker)"

    def __init__(self, master, controller):
        super().__init__(master)
        self.title("Settings")
        self.geometry("440x320")
        self.resizable(False, False)
        self.grab_set()  # Make modal — blocks interaction with main window

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
        self.grid_columnconfigure(1, weight=1)

        # Section: Microphone
        ctk.CTkLabel(self, text="Microphone", font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 4)
        )
        ctk.CTkLabel(self, text="Input device:").grid(
            row=1, column=0, sticky="w", padx=16, pady=4
        )
        self._mic_var = ctk.StringVar(value=current_label)
        self._mic_dropdown = ctk.CTkOptionMenu(
            self,
            values=self._dropdown_values,
            variable=self._mic_var,
            width=260,
        )
        self._mic_dropdown.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=4)

        # Section: Speech
        ctk.CTkLabel(self, text="Robot Speech", font=("", 13, "bold")).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 4)
        )

        # Speed slider
        ctk.CTkLabel(self, text="Speed:").grid(row=3, column=0, sticky="w", padx=16, pady=4)
        speed_frame = ctk.CTkFrame(self, fg_color="transparent")
        speed_frame.grid(row=3, column=1, sticky="ew", padx=(0, 16), pady=4)
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
        ctk.CTkLabel(self, text="Volume:").grid(row=4, column=0, sticky="w", padx=16, pady=4)
        vol_frame = ctk.CTkFrame(self, fg_color="transparent")
        vol_frame.grid(row=4, column=1, sticky="ew", padx=(0, 16), pady=4)
        vol_frame.grid_columnconfigure(0, weight=1)

        # Default volume from settings (add SPEECH_VOLUME to settings if desired; fallback 80)
        self._vol_var = ctk.IntVar(value=getattr(settings, 'SPEECH_VOLUME', 80))
        self._vol_slider = ctk.CTkSlider(
            vol_frame, from_=0, to=100, number_of_steps=100,
            variable=self._vol_var,
            command=lambda v: self._vol_label.configure(text=str(int(v)))
        )
        self._vol_slider.grid(row=0, column=0, sticky="ew")
        self._vol_label = ctk.CTkLabel(vol_frame, text=str(self._vol_var.get()), width=36)
        self._vol_label.grid(row=0, column=1, padx=(8, 0))

        # ── Apply / Close buttons ──
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(20, 12))

        ctk.CTkButton(btn_frame, text="Apply", width=120, command=self._on_apply).pack(
            side="left", padx=8
        )
        ctk.CTkButton(
            btn_frame, text="Close", width=100,
            fg_color="gray40", hover_color="gray30",
            command=self.destroy
        ).pack(side="left", padx=8)

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

        self._controller.apply_settings(
            mic_device_index=mic_device_index,
            mic_source=mic_source,
            speech_speed=speech_speed,
            volume=volume,
        )

        # Update in-memory defaults so sliders are correct if panel is re-opened
        settings.SPEECH_SPEED = speech_speed
        settings.SPEECH_VOLUME = volume