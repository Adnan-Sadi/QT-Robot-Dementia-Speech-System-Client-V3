from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional, Dict, Any

@dataclass
class Event:
    kind: str          # "stt_interim", "stt_final", "llm_response", "status", "error"
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)  # for future extensibility (emotions, movements, etc.)

class EventBus:
    def __init__(self):
        self._q: Queue = Queue()

    def publish(self, kind: str, text: str = "", **data):
        self._q.put(Event(kind=kind, text=text, data=data))

    def try_get(self) -> Optional[Event]:
        try:
            return self._q.get_nowait()
        except Empty:
            return None