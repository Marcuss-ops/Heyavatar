from typing import List
from contracts.gesture_planner import GesturePlanner, GestureIntent

class RuleBasedGesturePlanner(GesturePlanner):
    def plan(self, text: str, avatar_id: str, language: str) -> List[GestureIntent]:
        intents = []
        words = text.lower().split()
        
        # Heuristic keywords mapping to registry gestures
        if any(w in words for w in ["tre", "three"]):
            intents.append(GestureIntent(
                text_span="tre elementi fondamentali",
                gesture_id="count_three",
                anchor_word="tre",
                intensity=0.8
            ))
        elif any(w in words for w in ["due", "two"]):
            intents.append(GestureIntent(
                text_span="due cose importanti",
                gesture_id="count_two",
                anchor_word="due",
                intensity=0.7
            ))
        else:
            intents.append(GestureIntent(
                text_span="benvenuti",
                gesture_id="open_palms",
                anchor_word="benvenuti",
                intensity=0.5
            ))
        return intents
