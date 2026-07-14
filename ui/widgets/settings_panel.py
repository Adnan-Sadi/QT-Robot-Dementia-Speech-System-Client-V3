import customtkinter as ctk

from config.settings import settings
from services.stt_accumulator import STTAccumulator
from config.user_settings import save_user_settings


class SettingsPanel(ctk.CTkFrame):
    """
    An always-visible settings panel displayed on the left side of the main window.
    Allows adjusting microphone, speech speed, volume, and text size.
    Speed, volume, and font size changes are applied live (no Apply button needed).
    Microphone changes require pressing 'Apply Microphone' to take effect.
    """

    # Label shown in the dropdown for the robot's built-in ReSpeaker mic
    _ROS_MIC_LABEL = "QT Robot built-in mic (ReSpeaker)"

    def __init__(self, master, controller, main_window):
        super().__init__(master, fg_color=("gray90", "gray17"), corner_radius=8)
        self._controller = controller
        self._main_window = main_window  # Direct reference to MainWindow for callbacks

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

        # ── Section: Microphone ──
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
        self._mic_dropdown.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))

        # Apply button only for microphone (changing mic requires stream restart)
        ctk.CTkButton(
            self, text="Apply Microphone", width=180,
            command=self._on_apply_mic
        ).grid(row=4, column=0, padx=16, pady=(4, 12), sticky="w")

        # ── Section: Robot Speech ──
        ctk.CTkLabel(self, text="Robot Speech", font=("", 12, "bold")).grid(
            row=5, column=0, sticky="w", padx=16, pady=(8, 2)
        )

        # Speed slider (live)
        ctk.CTkLabel(self, text="Speed:").grid(row=6, column=0, sticky="w", padx=16, pady=2)
        self._speed_var = ctk.IntVar(value=settings.SPEECH_SPEED)
        self._make_slider_row(
            parent_row=7,
            var=self._speed_var,
            from_=50, to=200,
            number_of_steps=150,
            step=10,
            on_change=self._on_speed_change,
        )

        # Volume slider (live)
        ctk.CTkLabel(self, text="Volume:").grid(row=8, column=0, sticky="w", padx=16, pady=2)
        self._vol_var = ctk.IntVar(value=getattr(settings, 'SPEECH_VOLUME', 80))
        self._make_slider_row(
            parent_row=9,
            var=self._vol_var,
            from_=0, to=100,
            number_of_steps=100,
            step=10,
            on_change=self._on_volume_change,
        )

        # ── Section: Text Size ──
        ctk.CTkLabel(self, text="Text Size", font=("", 12, "bold")).grid(
            row=10, column=0, sticky="w", padx=16, pady=(8, 2)
        )
        ctk.CTkLabel(self, text="Font size:").grid(row=11, column=0, sticky="w", padx=16, pady=2)
        self._font_size_var = ctk.IntVar(value=getattr(settings, 'TRANSCRIPT_FONT_SIZE', 13))
        self._make_slider_row(
            parent_row=12,
            var=self._font_size_var,
            from_=10, to=28,
            number_of_steps=18,
            step=2,
            on_change=self._on_font_size_change,
        )

    # ------------------------------------------------------------------
    # Slider row builder
    # ------------------------------------------------------------------

    def _make_slider_row(self, parent_row, var, from_, to, number_of_steps, step, on_change):
        """
        Build a slider row with a − button on the left, slider in the middle,
        value label, and + button on the right. All packed into a sub-frame
        placed at parent_row in the settings panel grid.
        """
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=parent_row, column=0, sticky="ew", padx=16, pady=(0, 8))
        frame.grid_columnconfigure(1, weight=1)  # slider column expands

        # − button
        ctk.CTkButton(
            frame, text="−", width=28, height=28,
            command=lambda: self._step_slider(var, -step, from_, to, on_change)
        ).grid(row=0, column=0, padx=(0, 4))

        # Slider
        slider = ctk.CTkSlider(
            frame, from_=from_, to=to, number_of_steps=number_of_steps,
            variable=var,
            command=on_change,
        )
        slider.grid(row=0, column=1, sticky="ew")

        # Value label
        label = ctk.CTkLabel(frame, text=str(var.get()), width=36)
        label.grid(row=0, column=2, padx=(6, 4))

        # + button
        ctk.CTkButton(
            frame, text="+", width=28, height=28,
            command=lambda: self._step_slider(var, +step, from_, to, on_change)
        ).grid(row=0, column=3, padx=(4, 0))

        # Store label reference on var so the on_change callbacks can update it
        var._label = label

    def _step_slider(self, var, delta, from_, to, on_change):
        """Increment or decrement a slider variable by delta, clamped to [from_, to]."""
        new_val = max(from_, min(to, var.get() + delta))
        var.set(new_val)
        on_change(new_val)

    # ------------------------------------------------------------------
    # Live change callbacks
    # ------------------------------------------------------------------

    def _on_speed_change(self, value):
        """Apply speech speed immediately as the slider moves."""
        speed = int(value)
        self._speed_var._label.configure(text=str(speed))
        settings.SPEECH_SPEED = speed
        self._controller.apply_settings(
            mic_device_index=None,
            mic_source=None,
            speech_speed=speed,
            volume=None,
        )

    def _on_volume_change(self, value):
        """Apply volume immediately as the slider moves."""
        volume = int(value)
        self._vol_var._label.configure(text=str(volume))
        settings.SPEECH_VOLUME = volume
        self._controller.apply_settings(
            mic_device_index=None,
            mic_source=None,
            speech_speed=None,
            volume=volume,
        )

    def _on_font_size_change(self, value):
        """Apply font size immediately as the slider moves."""
        size = int(value)
        self._font_size_var._label.configure(text=str(size))
        settings.TRANSCRIPT_FONT_SIZE = size
        # Use the direct MainWindow reference — avoids the CTkFrame master chain issue
        self._main_window.set_transcript_font_size(size)

    # ------------------------------------------------------------------
    # Microphone apply (requires stream restart — kept behind a button)
    # ------------------------------------------------------------------

    def _on_apply_mic(self):
        """Read the microphone dropdown and push the change to the controller."""
        selected = self._mic_var.get()

        if selected == self._ROS_MIC_LABEL:
            mic_source = "default"
            mic_device_index = None
            mic_device_name = None
        else:
            # Parse the index out of "[N] Name (Hz)"
            mic_source = "external"
            mic_device_index = None
            mic_device_name = None
            try:
                mic_device_index = int(selected.split("]")[0].lstrip("["))
                # Look up the full device name by index for reliable future matching
                match = next((d for d in self._devices if d["index"] == mic_device_index), None)
                if match:
                    mic_device_name = match["name"]
            except (ValueError, IndexError):
                mic_device_index = None

        self._controller.apply_settings(
            mic_device_index=mic_device_index,
            mic_source=mic_source,
            speech_speed=None,
            volume=None,
        )
        # Update in-memory settings (including device name for future index resolution)
        settings.MIC_SOURCE = mic_source
        settings.MIC_DEVICE_INDEX = mic_device_index
        settings.MIC_DEVICE_NAME = mic_device_name

    def save_current_settings(self):
        """Persist the current in-memory settings to disk. Called on window close."""
        save_user_settings(settings)

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