from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class GestureIntent(BaseModel):
    text_span: str
    gesture_id: str
    anchor_word: str
    intensity: float

class GesturePlanner(ABC):
    @abstractmethod
    def plan(self, text: str, avatar_id: str, language: str) -> List[GestureIntent]:
        """Analyze text to generate gesture intents aligned to key words."""
        pass
