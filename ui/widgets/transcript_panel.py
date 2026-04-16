import customtkinter as ctk


class TranscriptPanel(ctk.CTkFrame):
    """Scrollable chat transcript with styled user/assistant messages."""

    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._textbox = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("", 13))
        self._textbox.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    def append_user(self, text):
        self._append(f"You:  {text}\n\n")

    def append_assistant(self, text):
        self._append(f"QT:  {text}\n\n")

    def append_system(self, text):
        self._append(f"   ── {text} ──\n\n")

    def clear(self):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.configure(state="disabled")

    def _append(self, text):
        self._textbox.configure(state="normal")
        self._textbox.insert("end", text)
        self._textbox.see("end")
        self._textbox.configure(state="disabled")
