import customtkinter as ctk
from config.settings import settings


class TranscriptPanel(ctk.CTkFrame):
    """Scrollable chat transcript showing only the most recent robot (assistant) response."""

    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._font_size = max(
            10, min(50, int(getattr(settings, 'TRANSCRIPT_FONT_SIZE', 13)))
        )
        self._font_size_var = ctk.IntVar(value=self._font_size)

        # Header with compact text-size controls in the top-right corner
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Transcript", font=("", 16, "bold")
        ).grid(row=0, column=0, sticky="w")

        controls = ctk.CTkFrame(header, fg_color="transparent")
        controls.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            controls, text="Text size", font=("", 14)
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            controls, text="−", width=28, height=28,
            command=lambda: self._step_font_size(-2),
        ).pack(side="left", padx=(0, 4))

        self._font_size_slider = ctk.CTkSlider(
            controls,
            from_=10, to=50, number_of_steps=40,
            width=110, height=16,
            variable=self._font_size_var,
            command=self._on_font_size_change,
        )
        self._font_size_slider.pack(side="left", padx=4)

        self._font_size_label = ctk.CTkLabel(
            controls, text=str(self._font_size), width=30, font=("", 13)
        )
        self._font_size_label.pack(side="left", padx=4)

        ctk.CTkButton(
            controls, text="+", width=28, height=28,
            command=lambda: self._step_font_size(2),
        ).pack(side="left", padx=(4, 0))

        self._textbox = ctk.CTkTextbox(
            self, wrap="word", state="disabled",
            font=("", self._font_size),
        )
        self._textbox.grid(
            row=1, column=0, sticky="nsew", padx=4, pady=4
        )
        self.set_font_size(self._font_size)

    def append_assistant(self, text):
        # Replace the entire content with only the latest response
        self._set(f"QT:  {text}\n\n")

    def append_system(self, text):
        self._set(f"   ── {text} ──\n\n")

    def clear(self):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")

    def set_font_size(self, size: int):
        """Update transcript text, controls, and the in-memory preference."""
        size = max(10, min(50, int(size)))
        self._font_size = size
        self._font_size_var.set(size)
        self._font_size_slider.set(size)
        self._font_size_label.configure(text=str(size))
        self._textbox.configure(font=("", size))
        settings.TRANSCRIPT_FONT_SIZE = size

    def _on_font_size_change(self, value):
        """Apply the slider value without writing settings on every move."""
        self.set_font_size(int(value))

    def _step_font_size(self, delta):
        """Use the same two-point steps as the previous + and − buttons."""
        self.set_font_size(self._font_size + delta)

    def _set(self, text):
        """Replace all content with the given text (shows only the most recent response)."""
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("end", text)
        self._textbox.see("1.0") # Scroll to the top
        self._textbox.configure(state="disabled")