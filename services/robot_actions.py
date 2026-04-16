import sys
import threading
import random
import rospy
from qt_robot_interface.srv import (
    speech_say, speech_config, speech_configRequest,
    behavior_talk_text, emotion_show
)
from qt_robot_interface import srv
from qt_gesture_controller.srv import gesture_play

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
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Initialize ROS node and all service proxies. Call once at startup."""
        if self._initialized:
            rospy.loginfo("RobotActions already initialized.")
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

            self._speech_say_service = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
            self._speech_config_service = rospy.ServiceProxy('/qt_robot/speech/config', speech_config)
            self._behavior_talk_service = rospy.ServiceProxy('/qt_robot/behavior/talkText', srv.behavior_talk_text)
            self._emotion_show_service = rospy.ServiceProxy('/qt_robot/emotion/show', srv.emotion_show)
            self._gesture_play_service = rospy.ServiceProxy('/qt_robot/gesture/play', gesture_play)

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
            req = speech_configRequest()
            req.language = ""
            req.pitch = 0
            req.speed = speed
            self._speech_config_service(req)
            rospy.loginfo(f"Speech speed set to {speed}.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Speech config failed: {e}")

    def say(self, text, emotion="neutral"):
        """
        Make the robot speak with a matching gesture.
        Blocks until speech is complete - the controller uses this to know
        when to resume listening.
        """
        if not self._behavior_talk_service:
            rospy.logerr("Speech service not initialized.")
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