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


def main():
    # 1. Initialize ROS and robot services
    robot = RobotActions()
    robot.initialize()
    robot.configure_speech_speed(settings.SPEECH_SPEED)

    # 2. Create shared services
    bus = EventBus()
    backend = BackendBridge()
    stt = STTAccumulator(bus, backend=backend)

    # 3. Create controller (orchestrates everything)
    controller = ChatController(bus, robot, stt, backend)

    # 4. Launch UI (blocks on mainloop)
    win = MainWindow(controller, bus)
    win.mainloop()

    # 5. Cleanup on exit
    if controller.is_session_active():
        controller.stop_session()


if __name__ == "__main__":
    main()