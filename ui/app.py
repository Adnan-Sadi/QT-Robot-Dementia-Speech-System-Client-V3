import customtkinter as ctk
from ui.widgets.transcript_panel import TranscriptPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.settings_panel import SettingsPanel
from config.settings import settings


class MainWindow(ctk.CTk):
    def __init__(self, controller, bus):
        super().__init__()
        self.title("QT Robot Speech System Client")
        self.geometry("1024x768")
        self.minsize(700, 450)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._controller = controller
        self._bus = bus

        # ── Top-level grid: header row, main content row, send button row, status bar row ──
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header toolbar ──
        toolbar = ctk.CTkFrame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        self._start_btn = ctk.CTkButton(
            toolbar, text="▶  Start Chat", width=200, height=50, font=("", 18, "bold"),
            fg_color="#2B7A0B", hover_color="#1E5C08",
            command=self._on_start
        )
        # place in center
        self._start_btn.pack(side="left", padx=6, pady=8)

        self._stop_btn = ctk.CTkButton(
            toolbar, text="■  Stop Chat", width=200, height=50, font=("", 18, "bold"),
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

        # ── Send button (centred, large) ──
        send_frame = ctk.CTkFrame(self, fg_color="transparent")
        send_frame.grid(row=2, column=0, pady=(4, 8))

        self._send_btn = ctk.CTkButton(
            send_frame, text="Send", width=220, height=55,
            font=("", 20, "bold"),
            fg_color="#9333EA",             # Bright purple background
            hover_color="#7E22CE",          # Slightly darker purple when hovered
            text_color="#FFFFFF",           # Bright white font for contrast
            text_color_disabled="#D8B4FE",  # Light purple font when the button is disabled
            command=self._on_send, 
            state="disabled"
        )
        self._send_btn.pack()

        # ── Status bar ──
        self._status = StatusBar(self)
        self._status.grid(row=3, column=0, sticky="ew")

        # Start event polling
        self._poll_bus()

        # Save settings when the window is closed
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self):
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        # self._send_btn.configure(state="normal")
        self._settings.set_session_active(True)
        self._controller.start_session()

    def _on_stop(self):
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._send_btn.configure(state="disabled")
        self._settings.set_session_active(False)
        self._controller.stop_session()

    def _on_send(self):
        self._send_btn.configure(state="disabled")
        self._controller.send_message()

    def set_transcript_font_size(self, size: int):
        """Called by SettingsPanel when the font size slider is moved."""
        self._transcript.set_font_size(size)
        # Only update the in-memory value here; saving is deferred to window close / Apply
        settings.TRANSCRIPT_FONT_SIZE = size

    def _on_window_close(self):
        """Called when the user closes the window. Saves settings before exiting."""
        self._settings.save_current_settings()
        self.destroy()

    # ------------------------------------------------------------------
    # Close-session countdown
    # ------------------------------------------------------------------

    def _begin_close_countdown(self, seconds_remaining=5):
        """Open a centered overlay window showing the countdown, then destroy the app."""
        overlay = ctk.CTkToplevel(self)
        overlay.title("")
        overlay.resizable(False, False)
        overlay.grab_set()  # make it modal so user can't interact with the main window

        # Square size
        size = 300
        overlay.geometry(f"{size}x{size}")
        # Center over the main window
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - size) // 2
        y = self.winfo_y() + (self.winfo_height() - size) // 2
        overlay.geometry(f"{size}x{size}+{x}+{y}")

        # "Session complete" label at the top
        ctk.CTkLabel(
            overlay,
            text="Session complete.\nWindow closing in",
            font=("", 16, "bold"),
            justify="center"
        ).pack(pady=(30, 10))

        # Canvas for the circular timer
        canvas = ctk.CTkCanvas(overlay, width=140, height=140, bg="#2b2b2b", highlightthickness=0)
        canvas.pack()

        def _draw_circle(n):
            canvas.delete("all")
            # Outer circle
            canvas.create_oval(10, 10, 130, 130, outline="#9333EA", width=6, fill="#1a1a2e")
            # Number in the center
            canvas.create_text(70, 70, text=str(n), fill="white", font=("", 48, "bold"))

        def _tick(n):
            if n > 0:
                _draw_circle(n)
                overlay.after(1000, lambda: _tick(n - 1))
            else:
                overlay.destroy()
                self.destroy()

        _tick(seconds_remaining)

    # ------------------------------------------------------------------
    # Event bus polling
    # ------------------------------------------------------------------

    def _poll_bus(self):
        """Poll the event bus and update UI accordingly."""
        ev = self._bus.try_get()
        while ev:
            kind = ev.kind

            if kind == "llm_response":
                # Only show what the robot said — no user text, no scenario label
                self._transcript.append_assistant(ev.text)

            elif kind == "status":
                self._status.set(ev.text)
                # Enable/disable the Send button based on status
                if ev.text == "Listening...":
                    self._send_btn.configure(state="normal")
                elif ev.text in ("Thinking...", "Speaking..."):
                    self._send_btn.configure(state="disabled")

            elif kind == "error":
                self._transcript.append_system(f"⚠ {ev.text}")

            elif kind == "chat_ended":
                # Robot has finished its final utterance — reset UI state and begin countdown
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled")
                self._send_btn.configure(state="disabled")
                self._settings.set_session_active(False)
                self._begin_close_countdown(seconds_remaining=5)

            ev = self._bus.try_get()

        self.after(50, self._poll_bus)