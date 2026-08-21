import customtkinter as ctk
from config.settings import settings 


class TranscriptPanel(ctk.CTkFrame):
    """Scrollable chat transcript showing only the most recent robot (assistant) response."""

    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._font_size = getattr(settings, 'TRANSCRIPT_FONT_SIZE', 13)
        self._textbox = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("", self._font_size))
        self._textbox.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

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
        """Update the font size of the transcript text box."""
        self._font_size = size
        self._textbox.configure(font=("", self._font_size))

    def _set(self, text):
        """Replace all content with the given text (shows only the most recent response)."""
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("end", text)
        self._textbox.see("1.0") # Scroll to the top 
        self._textbox.configure(state="disabled")