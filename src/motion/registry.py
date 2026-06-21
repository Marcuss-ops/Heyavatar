import yaml
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel

class GestureEntry(BaseModel):
    id: str
    name: str
    duration_seconds: float

class GestureRegistry:
    def __init__(self, registry_file: Path = Path("registry/gestures.yaml")):
        self.registry_file = registry_file
        self.gestures: Dict[str, GestureEntry] = {}
        self.load()

    def load(self):
        if not self.registry_file.is_file():
            return
        with open(self.registry_file, "r") as f:
            data = yaml.safe_load(f)
            if not data or "gestures" not in data:
                return
            for item in data["gestures"]:
                entry = GestureEntry(**item)
                self.gestures[entry.id] = entry

    def get_gesture(self, gesture_id: str) -> GestureEntry:
        if gesture_id not in self.gestures:
            raise KeyError(f"Gesture {gesture_id} not registered.")
        return self.gestures[gesture_id]

    def list_gestures(self) -> List[GestureEntry]:
        return list(self.gestures.values())
