import customtkinter as ctk


class StatusBar(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, height=32, fg_color="transparent")
        self._indicator = ctk.CTkLabel(self, text="●", text_color="gray", font=("", 12))
        self._indicator.pack(side="left", padx=(8, 4))
        self._label = ctk.CTkLabel(self, text="Ready", font=("", 12))
        self._label.pack(side="left", padx=4)

    def set(self, text):
        self._label.configure(text=text)
        # Color the indicator based on status
        if "Listening" in text:
            self._indicator.configure(text_color="#22C55E")  # green
        elif "Speaking" in text or "Thinking" in text:
            self._indicator.configure(text_color="#F59E0B")  # amber
        elif "error" in text.lower():
            self._indicator.configure(text_color="#EF4444")  # red
        else:
            self._indicator.configure(text_color="gray")
