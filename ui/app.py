import customtkinter as ctk
from ui.widgets.transcript_panel import TranscriptPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.settings_panel import SettingsPanel


class MainWindow(ctk.CTk):
    def __init__(self, controller, bus):
        super().__init__()
        self.title("QT Robot Agentic Speech System Client")
        self.geometry("1024x768")
        self.minsize(700, 450)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._controller = controller
        self._bus = bus

        # ── Top-level grid: header row, main content row, status bar row ──
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header toolbar ──
        toolbar = ctk.CTkFrame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        self._start_btn = ctk.CTkButton(
            toolbar, text="▶  Start Chat", width=140,
            fg_color="#2B7A0B", hover_color="#1E5C08",
            command=self._on_start
        )
        self._start_btn.pack(side="left", padx=6, pady=8)

        self._stop_btn = ctk.CTkButton(
            toolbar, text="■  Stop Chat", width=140,
            fg_color="#B91C1C", hover_color="#7F1D1D",
            command=self._on_stop, state="disabled"
        )
        self._stop_btn.pack(side="left", padx=6, pady=8)

        # ── Main content area: settings on left, transcript on right ──
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        # Left column (settings) is fixed width; right column (transcript) expands
        content_frame.grid_columnconfigure(0, weight=0)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Settings panel (always visible, left side)
        self._settings = SettingsPanel(content_frame, self._controller, main_window=self)
        self._settings.grid(row=0, column=0, sticky="ns", padx=(0, 8), pady=0)

        # Transcript panel (right side, expands to fill space)
        self._transcript = TranscriptPanel(content_frame)
        self._transcript.grid(row=0, column=1, sticky="nsew")

        # self._transcript.append_assistant("Hello! I'm QT. How are you feeling today?")
        # self._transcript.append_assistant("That's wonderful to hear! Would you like to talk about something?")

        # ── Input bar (transcript preview + Send button) ──
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        input_frame.grid_columnconfigure(0, weight=1)

        self._transcript_preview = ctk.CTkLabel(
            input_frame,
            text="(Start chat and speak",
            anchor="w",
            wraplength=600,
            text_color="gray",
            font=("", 13),
        )
        self._transcript_preview.grid(row=0, column=0, sticky="ew", padx=12, pady=8)

        self._send_btn = ctk.CTkButton(
            input_frame, text="Send", width=120,
            font=("", 14, "bold"),
            command=self._on_send, state="disabled"
        )
        self._send_btn.grid(row=0, column=1, padx=8, pady=8)

        # ── Status bar ──
        self._status = StatusBar(self)
        self._status.grid(row=3, column=0, sticky="ew")

        # Start event polling
        self._poll_bus()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self):
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._send_btn.configure(state="normal")
        self._controller.start_session()

    def _on_stop(self):
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._send_btn.configure(state="disabled")
        self._transcript_preview.configure(text="(Session ended)", text_color="gray")
        self._controller.stop_session()

    def _on_send(self):
        self._send_btn.configure(state="disabled")
        self._controller.send_message()

    def set_transcript_font_size(self, size: int):
        """Called by SettingsPanel when the font size slider is moved."""
        self._transcript.set_font_size(size)

    # ------------------------------------------------------------------
    # Event bus polling
    # ------------------------------------------------------------------

    def _poll_bus(self):
        """Poll the event bus and update UI accordingly."""
        ev = self._bus.try_get()
        while ev:
            kind = ev.kind

            if kind == "stt_interim":
                # Show live interim transcript in italic/gray
                self._transcript_preview.configure(text=ev.text, text_color="gray")

            elif kind == "stt_final":
                if ev.text:
                    # Show accumulated final text — ready to send
                    self._transcript_preview.configure(text=ev.text, text_color="white")
                    self._send_btn.configure(state="normal")
                else:
                    self._transcript_preview.configure(
                        text="(Listening...)", text_color="gray"
                    )

            elif kind == "user_message":
                # User message is no longer shown in the transcript panel;
                # only the transcript preview bar (input bar) is used for user speech display.
                self._transcript_preview.configure(text="(Waiting for response...)", text_color="gray")

            elif kind == "llm_response":
                # Only show what the robot said — no user text, no scenario label
                self._transcript.append_assistant(ev.text)
                # Re-enable send after robot finishes speaking
                self._send_btn.configure(state="normal")

            elif kind == "status":
                self._status.set(ev.text)

            elif kind == "error":
                self._transcript.append_system(f"⚠ {ev.text}")
                self._send_btn.configure(state="normal")

            ev = self._bus.try_get()

        self.after(50, self._poll_bus)