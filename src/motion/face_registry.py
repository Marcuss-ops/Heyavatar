from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


class FaceMotionEntry(BaseModel):
    id: str
    name: str
    duration_seconds: float
    intensity: float = 0.2
    family: str = "expression"
    tags: List[str] = Field(default_factory=list)
    requires_hands: bool = False
    notes: str = ""


class FaceMotionRegistry:
    def __init__(self, registry_file: Path = Path("registry/facial_motions.yaml")):
        self.registry_file = registry_file
        self.motions: Dict[str, FaceMotionEntry] = {}
        self.load()

    def load(self) -> None:
        self.motions.clear()
        if not self.registry_file.is_file():
            return
        with self.registry_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for item in data.get("motions", []) or []:
            entry = FaceMotionEntry(**item)
            self.motions[entry.id] = entry

    def get_motion(self, motion_id: str) -> FaceMotionEntry:
        if motion_id not in self.motions:
            raise KeyError(f"Face motion {motion_id} not registered.")
        return self.motions[motion_id]

    def list_motions(self) -> List[FaceMotionEntry]:
        return list(self.motions.values())

    def list_hand_free(self) -> List[FaceMotionEntry]:
        return [motion for motion in self.motions.values() if not motion.requires_hands]

