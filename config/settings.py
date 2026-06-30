import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    # Backend Configuration
    BASE_HTTP_URL = os.getenv("BASE_HTTP_URL", "https://cognibot.org")
    WS_PATH = os.getenv("WS_PATH", "/ws/chat/")
    SOURCE = os.getenv("SOURCE", "qtrobot")
    USERNAME = os.getenv("USERNAME")
    PASSWORD = os.getenv("PASSWORD")
    
    # Audio Configuration
    AUDIO_RATE = int(os.getenv("AUDIO_RATE", "16000"))
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-US")
    SPEECH_MODEL = os.getenv("SPEECH_MODEL", "default")
    USE_ENHANCED_MODEL = os.getenv("USE_ENHANCED_MODEL", "True").lower() == "true"

    # Microphone source: "default" (ReSpeaker ROS topic) or "external" (USB mic)
    MIC_SOURCE = os.getenv("MIC_SOURCE", "default").lower()
    MIC_DEVICE_INDEX = os.getenv("MIC_DEVICE_INDEX", None)
    
    # Speech Configuration
    SPEECH_SPEED = int(os.getenv("SPEECH_SPEED", "90"))
    SPEECH_VOLUME = int(os.getenv("SPEECH_VOLUME", "80"))
    GREETING_TEXT  = os.getenv("GREETING_TEXT", "Hello! How are you feeling today?")
    
    # Timeout Configuration
    DEFAULT_TIMEOUT = float(os.getenv("DEFAULT_TIMEOUT", "20.0"))
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25.0"))
    
    # Emotion Configuration
    EMOTION_LISTENING = os.getenv("EMOTION_LISTENING", "QT/confused,QT/showing_smile").split(",")

settings = Settings()
