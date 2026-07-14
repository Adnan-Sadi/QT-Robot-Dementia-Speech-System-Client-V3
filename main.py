#!/usr/bin/env python3
"""
QT Robot Speech System V3 — Entry point.
Runs ROS init, backend, STT, and UI all in one process.
"""
from services.event_bus import EventBus
from services.backend_client import BackendBridge
from services.stt_accumulator import STTAccumulator
from services.robot_actions import RobotActions
from controllers.chat_controller import ChatController
from ui.app import MainWindow
from config.settings import settings
from config.user_settings import load_user_settings


def main():
    # Load last-saved user preferences (overrides .env defaults)
    load_user_settings(settings)

    # Initialize ROS and robot services
    robot = RobotActions()
    robot.initialize()
    robot.configure_speech_speed(settings.SPEECH_SPEED)

    # Create shared services
    bus = EventBus()
    backend = BackendBridge()
    stt = STTAccumulator(bus, backend=backend)

    #  Create controller (orchestrates everything)
    controller = ChatController(bus, robot, stt, backend)

    #  Launch UI (blocks on mainloop)
    win = MainWindow(controller, bus)
    win.mainloop()

    # Cleanup on exit
    if controller.is_session_active():
        controller.stop_session()


if __name__ == "__main__":
    main()