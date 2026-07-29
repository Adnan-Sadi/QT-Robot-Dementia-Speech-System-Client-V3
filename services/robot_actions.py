import sys
import time
import threading
import random
#try:
import rospy
from qt_robot_interface.srv import (
    speech_say, speech_config, speech_configRequest, setting_setVolume, setting_setVolumeRequest,
    behavior_talk_text, emotion_show
)
from qt_robot_interface import srv
from qt_gesture_controller.srv import gesture_play
ROS_AVAILABLE = True
# except ImportError:
#     ROS_AVAILABLE = False

from config.settings import settings


class RobotActions:
    """
    Encapsulates all QT Robot physical actions: speech, gestures, emotions.
    """

    def __init__(self):
        self._speech_say_service = None
        self._speech_config_service = None
        self._behavior_talk_service = None
        self._emotion_show_service = None
        self._gesture_play_service = None
        self._set_volume_service = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Initialize ROS node and all service proxies. Call once at startup."""
        if self._initialized:
            print("RobotActions already initialized.")
            return
        if not ROS_AVAILABLE:
            print("[RobotActions] ROS not available — running in UI-only mode, robot actions disabled.")
            self._initialized = True
            return
            
        try:
            rospy.init_node('qt_agentic_speech_system', anonymous=True)
            rospy.loginfo("ROS node 'qt_agentic_speech_system' started.")

            rospy.loginfo("Waiting for QT Robot services...")
            rospy.wait_for_service('/qt_robot/speech/say')
            rospy.wait_for_service('/qt_robot/speech/config')
            rospy.wait_for_service('/qt_robot/behavior/talkText')
            rospy.wait_for_service('/qt_robot/emotion/show')
            rospy.wait_for_service('/qt_robot/gesture/play')
            rospy.wait_for_service('/qt_robot/setting/setVolume')

            self._speech_say_service = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
            self._speech_config_service = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)
            self._behavior_talk_service = rospy.ServiceProxy('/qt_robot/behavior/talkText', srv.behavior_talk_text)
            self._emotion_show_service = rospy.ServiceProxy('/qt_robot/emotion/show', srv.emotion_show)
            self._gesture_play_service = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)
            self._set_volume_service = rospy.ServiceProxy('/qt_robot/setting/setVolume', setting_setVolume,)

            rospy.loginfo("All QT Robot services available.")
            self._initialized = True

        except rospy.ROSException as e:
            rospy.logerr(f"Failed to initialize RobotActions: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    def configure_speech_speed(self, speed):
        """Set the robot's speech speed."""
        if not self._speech_config_service:
            return
        try:
            req = speech_configRequest(language='en-US', speed=speed, pitch=0)
            self._speech_config_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Speech config failed: {e}")

    def configure_volume(self, volume: int):
        """Set the robot's speaker volume (0–100)."""
        if not self._set_volume_service:
            return
        try:
            req = setting_setVolumeRequest(volume=volume)
            self._set_volume_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Set volume failed: {e}")

    def say(self, text, emotion="neutral"):
        """
        Make the robot speak with a matching gesture.
        Blocks until speech is complete - the controller uses this to know
        when to resume listening.
        """
        if not self._behavior_talk_service:
            print(f"Speech Service not initialized or say() was called (no-op in UI-only mode): '{text}'")
            return

        # Play gesture in background
        gesture_name = self._gesture_for_mood(emotion)
        if gesture_name:
            threading.Thread(target=self._play_gesture, args=(gesture_name,), daemon=True).start()

        # Speak (blocking)
        try:
            req = srv.behavior_talk_textRequest()
            req.message = text
            resp = self._behavior_talk_service(req)
            if not resp.status:
                rospy.logwarn("Speech service call returned failure status.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Speech service failed: {e}")

    # ------------------------------------------------------------------
    # Emotions & Gestures
    # ------------------------------------------------------------------

    def show_emotion(self, name):
        """Show an emotion on the robot's face."""
        if not self._emotion_show_service:
            return
        try:
            self._emotion_show_service(name)
        except Exception as e:
            rospy.logwarn(f"Emotion show failed: {e}")

    def play_gesture(self, name):
        """Play a gesture (non-blocking, runs in a thread)."""
        threading.Thread(target=self._play_gesture, args=(name,), daemon=True).start()

    def _play_gesture(self, name):
        if not self._gesture_play_service:
            return
        try:
            resp = self._gesture_play_service(name, 0)
            if resp.status:
                self._gesture_play_service("QT/neutral", 0)
        except Exception as e:
            rospy.logwarn(f"Gesture play failed: {e}")

    ## ------------------------------------------------------------------
    # Greeting The user at the start of the session
    ## ------------------------------------------------------------------
    def greet(self, greeting_text: str):
        """
        Play a wakeup gesture, show a happy emotion, then speak the greeting.
        Blocks until speech is complete, controller uses this to know when to start listening.
        Called once at the start of each session.
        """
        if not ROS_AVAILABLE:
            print(f"[RobotActions] greet() called (no-op in UI-only mode): '{greeting_text}'")
            return

        # Show yawn emotion first
        try:
            if self._emotion_show_service:
                self._emotion_show_service("QT/yawn")
        except Exception as e:
            print(f"[RobotActions] greet emotion failed: {e}")

        # Play wakeup arm gesture in background — starts while the yawn face is showing
        threading.Thread(target=self._play_gesture, args=("QT/happy",), daemon=True).start()

        # Small pause so the emotion animation has time to fully display before lips start moving
        time.sleep(2.0)

        # Speak the greeting (blocking — returns when robot finishes speaking)
        self.say(greeting_text, emotion="happy")

    # ------------------------------------------------------------------
    # Future: execute backend-commanded actions
    # ------------------------------------------------------------------

    def execute_actions(self, actions_dict):
        """
        Execute actions from backend response.
        
        Example of Expected format:
            {"emotion": "QT/happy", "gesture": "QT/wave", "movement": "..."}
        """
        if not actions_dict:
            return
        if "emotion" in actions_dict:
            self.show_emotion(actions_dict["emotion"])
        if "gesture" in actions_dict:
            self.play_gesture(actions_dict["gesture"])
        # movement handling (rather than just pre-recorded gestures) is something I might look into in future.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gesture_for_mood(mood):
        mapping = {
            "happy": lambda: random.choice(['approval', 'QT/point_front', 'QT/swipe_left', 'QT/swipe_right']),
            "sad": lambda: 'QT/sad',
            "surprised": lambda: 'QT/surprise',
            "angry": lambda: 'QT/angry',
            "scared": lambda: 'QT/peekaboo',
            "neutral": lambda: random.choice(['QT/neutral', 'QT/show_left', 'QT/show_right', 'QT/point_front']),
        }
        fn = mapping.get(mood, mapping["neutral"])
        return fn()