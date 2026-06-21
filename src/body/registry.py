import yaml
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel

class BodyProfileEntry(BaseModel):
    id: str
    name: str
    gender: str
    style: str
    outfits: List[str]
    cameras: List[str]

class RenderProfileEntry(BaseModel):
    id: str
    resolution: List[int]
    fps: int
    codec: str
    bitrate_bps: int

class BodyProfileRegistry:
    def __init__(self, body_yaml: Path = Path("registry/body_profiles.yaml"), render_yaml: Path = Path("registry/render_profiles.yaml")):
        self.body_yaml = body_yaml
        self.render_yaml = render_yaml
        self.bodies: Dict[str, BodyProfileEntry] = {}
        self.renders: Dict[str, RenderProfileEntry] = {}
        self.load()

    def load(self):
        if self.body_yaml.is_file():
            with open(self.body_yaml, "r") as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    for p in data["profiles"]:
                        self.bodies[p["id"]] = BodyProfileEntry(**p)
        if self.render_yaml.is_file():
            with open(self.render_yaml, "r") as f:
                data = yaml.safe_load(f)
                if data and "profiles" in data:
                    for r in data["profiles"]:
                        self.renders[r["id"]] = RenderProfileEntry(**r)
                        
    def get_body(self, body_id: str) -> BodyProfileEntry:
        return self.bodies[body_id]
        
    def get_render(self, render_id: str) -> RenderProfileEntry:
        return self.renders[render_id]
