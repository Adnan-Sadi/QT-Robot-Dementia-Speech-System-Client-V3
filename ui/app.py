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

        self._closing = False
        self._auto_closing = False
        self._listening = False
        self._error_active = False
        self._settings_visible = False
        self._transcript_visible = False
        self._last_control_state = None

        # ── Top-level grid: header row, main content row, status bar row ──
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header toolbar ──
        toolbar = ctk.CTkFrame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        self._start_btn = ctk.CTkButton(
            toolbar, text="▶  Start Chat", width=140, height=40,
            font=("", 16, "bold"), fg_color="#2B7A0B", hover_color="#1E5C08",
            command=self._on_start,
        )
        self._start_btn.pack(side="left", padx=6, pady=8)

        self._stop_btn = ctk.CTkButton(
            toolbar, text="■  Stop Chat", width=140, height=40,
            font=("", 16, "bold"), fg_color="#B91C1C", hover_color="#7F1D1D",
            command=self._on_stop, state="disabled",
        )
        self._stop_btn.pack(side="left", padx=6, pady=8)

        self._settings_btn = ctk.CTkButton(
            toolbar, text="Settings", width=140, height=40,
            font=("", 16), command=self._show_settings,
        )
        self._settings_btn.pack(side="right", padx=6, pady=8)

        # ── Main content area: conversation and settings screens ──
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        self._main_screen = ctk.CTkFrame(content_frame, fg_color="transparent")
        self._main_screen.grid(row=0, column=0, sticky="nsew")
        self._main_screen.grid_columnconfigure(0, weight=1)
         # Share available height between Send and the expanded transcript.
        self._main_screen.grid_rowconfigure(0, weight=3, uniform="conversation")
        self._main_screen.grid_rowconfigure(2, weight=0)

        self._settings_screen = ctk.CTkFrame(content_frame, fg_color="transparent")
        self._settings_screen.grid(row=0, column=0, sticky="nsew")
        self._settings_screen.grid_columnconfigure(0, weight=1)
        self._settings_screen.grid_rowconfigure(1, weight=1)

        self._back_btn = ctk.CTkButton(
            self._settings_screen, text="←  Back to conversation",
            width=210, height=40, command=self._show_main,
        )
        self._back_btn.grid(row=0, column=0, sticky="w", pady=(4, 8))

        self._settings = SettingsPanel(
            self._settings_screen, self._controller, main_window=self
        )
        self._settings.grid(row=1, column=0, sticky="nsew")

        # ── Send button (centred, large) ──
        send_frame = ctk.CTkFrame(self._main_screen, fg_color="transparent")
        send_frame.grid(row=0, column=0, sticky="nsew")
        # Center the Send button and its keyboard reminder together
        send_group = ctk.CTkFrame(send_frame, fg_color="transparent")
        send_group.place(relx=0.5, rely=0.5, anchor="center")

        self._send_btn = ctk.CTkButton(
            send_group, text="Send", width=180, height=180, corner_radius=90,
            font=("", 32, "bold"),
            fg_color="#9333EA",             # Bright purple background
            hover_color="#7E22CE",          # Slightly darker purple when hovered
            text_color="#FFFFFF",           # Bright white font for contrast
            text_color_disabled="#D8B4FE",  # Light purple font when the button is disabled
            command=self._on_send,
            state="disabled",
        )
        self._send_btn.grid(row=0, column=0)

        ctk.CTkLabel(
            send_group,
            text="Press Enter to send",
            font=("", 16),
            text_color=("gray35", "gray70"),
        ).grid(row=1, column=0, pady=(10, 0))

        send_frame.bind("<Configure>", self._resize_send)

        self._transcript = None
        self._transcript_btn = None
        if settings.ENABLE_TRANSCRIPT:
            self._transcript_btn = ctk.CTkButton(
                self._main_screen, text="Show transcript", width=170,
                fg_color="gray25", hover_color="gray35",
                command=self._toggle_transcript,
            )
            self._transcript_btn.grid(row=1, column=0, pady=(4, 8))

            self._transcript = TranscriptPanel(self._main_screen)
            self._transcript.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
            self._transcript.grid_propagate(False)
            self._transcript.grid_remove()

        self._main_screen.tkraise()

        # ── Status bar ──
        self._status = StatusBar(self)
        self._status.grid(row=2, column=0, sticky="ew")

        # Start event polling
        self._poll_bus()

        # Save settings when the window is closed
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Keyboard shortcuts for the conversation and settings screens
        for key in ("Return", "KP_Enter", "Up", "Down", "Left", "Right"):
            self.bind(f"<{key}>", self._on_main_key, add="+")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._controller.start_session():
            self._listening = False
            self._show_main()
        self._refresh_controls()

    def _on_stop(self):
        self._listening = False
        self._controller.stop_session()
        self._refresh_controls()

    def _on_send(self):
        if self._controller.send_message():
            self._listening = False
        self._refresh_controls()

    def _on_main_key(self, event):
        """Handle conversation shortcuts only on the main screen."""
        if self._closing or self._auto_closing or self.grab_current() is not None:
            return

        if event.keysym in ("Return", "KP_Enter"):
            # Only send if the Send button is enabled and the settings panel is not visible
            if self._send_btn.cget("state") == "normal" and not self._settings_visible:
                self._on_send()

        elif event.keysym == "Up":
            self._settings.adjust_volume(5)

        elif event.keysym == "Down":
            self._settings.adjust_volume(-5)

        elif event.keysym == "Right":
            self._settings.adjust_speed(5)

        elif event.keysym == "Left":
            self._settings.adjust_speed(-5)

        else:
            return

        return "break"

    def _show_settings(self):
        self._settings_visible = True
        self._settings_screen.tkraise()
        self.focus_set()
        self._refresh_controls()

    def _show_main(self):
        self._controller.stop_voice_preview()
        self._settings.save_current_settings()
        self._settings_visible = False
        self._main_screen.tkraise()
        self.focus_set()
        self._refresh_controls()

    def _resize_send(self, event):
        # Keep the circle visible when the transcript is open or the window is small.
        diameter = min(220, max(100, min(event.width - 24, event.height - 64)))
        self._send_btn.configure(
            width=diameter, height=diameter, corner_radius=diameter // 2,
            font=("", max(18, diameter // 7), "bold"),
        )

    def _toggle_transcript(self):
        if self._transcript is None:
            return
        self._transcript_visible = not self._transcript_visible
        if self._transcript_visible:
            self._main_screen.grid_rowconfigure(2, weight=2, uniform="conversation")
            self._transcript.grid()
            self._transcript_btn.configure(text="Hide transcript")
        else:
            self._transcript.grid_remove()
            self._main_screen.grid_rowconfigure(2, weight=0, uniform="")
            self._transcript_btn.configure(text="Show transcript")

    def _refresh_controls(self):
        active = self._controller.is_session_active()
        stopping = self._controller.is_session_stopping()
        preview = self._controller.is_voice_preview_active()
        preview_stopping = self._controller.is_voice_preview_stopping()
        busy = self._controller.is_busy()
        locked = self._closing or self._auto_closing
        state = (
            active, stopping, preview, preview_stopping, busy, locked,
            self._listening, self._settings_visible,
        )
        if state == self._last_control_state:
            return
        self._last_control_state = state

        self._start_btn.configure(state="normal" if not busy and not locked else "disabled")
        self._stop_btn.configure(state="normal" if active and not locked else "disabled")
        self._send_btn.configure(
            state="normal" if active and self._listening and not locked else "disabled"
        )
        self._settings_btn.configure(
            state="disabled" if self._settings_visible or locked else "normal"
        )
        self._back_btn.configure(state="disabled" if locked else "normal")
        self._settings.set_session_active(active or stopping)
        self._settings.set_voice_preview_state(
            active=preview,
            stopping=preview_stopping,
            available=not busy and not locked,
            locked=locked,
        )

    def set_transcript_font_size(self, size: int):
        """Called by SettingsPanel when the font size slider is moved."""
        if self._transcript is not None:
            self._transcript.set_font_size(size)
        # Only update the in-memory value here; saving is deferred to window close / Apply
        settings.TRANSCRIPT_FONT_SIZE = size

    def _on_window_close(self):
        """Called when the user closes the window. Saves settings before exiting."""
        if self._closing:
            return
        self._closing = True
        self._settings.save_current_settings()
        self._controller.stop_voice_preview()
        if self._controller.is_session_active():
            self._controller.stop_session()
        self._status.set("Finishing current activity...")
        self._refresh_controls()
        self._finish_close()

    def _finish_close(self):
        if self._controller.is_busy():
            self.after(100, self._finish_close)
        else:
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
                self._on_window_close()

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
                if self._transcript is not None:
                    self._transcript.append_assistant(ev.text)

            elif kind == "status":
                if not self._error_active or ev.text in (
                    "Connecting to backend...", "Listening...", "Thinking...", "Speaking..."
                ):
                    self._error_active = False
                    self._status.set(ev.text)
                # Enable/disable the Send button based on status
                self._listening = (
                    ev.text == "Listening..." and self._controller.is_session_active()
                )

            elif kind == "error":
                self._error_active = True
                self._status.set(f"Error: {ev.text}")
                if self._transcript is not None:
                    self._transcript.append_system(f"⚠ {ev.text}")

            elif kind == "voice_preview":
                self._error_active = ev.data.get("state") == "error"
                self._status.set(ev.text)
                if ev.data.get("state") in ("idle", "error"):
                    self._settings.save_current_settings()

            elif kind == "chat_ended":
                # Robot has finished its final utterance — reset UI state and begin countdown
                self._listening = False
                self._settings.save_current_settings()
                if not self._closing and not self._auto_closing:
                    self._auto_closing = True
                    self._refresh_controls()
                    self._begin_close_countdown(seconds_remaining=5)

            ev = self._bus.try_get()

        self._refresh_controls()
        self.after(50, self._poll_bus)
