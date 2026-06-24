from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.motion.face_registry import FaceMotionRegistry


@dataclass(slots=True, frozen=True)
class FaceMotionIntent:
    text_span: str
    motion_id: str
    anchor_word: str
    intensity: float


class RuleBasedFaceMotionPlanner:
    """Very small, hand-free motion planner for avatar facial cues.

    The planner intentionally keeps the motion vocabulary tiny so it
    can be reused as a lightweight default in code paths that do not
    want hand choreography at all.
    """

    _QUESTION_WORDS = {"?", "perché", "perche", "why", "what", "come", "how"}
    _EMPHASIS_WORDS = {
        "molto", "importante", "fondamentale", "chiave", "really", "important",
        "fantastic", "fantastico", "optimized", "ottimizzato", "great",
        "eccellente", "incredible", "incredibile", "amazing", "outstanding"
    }
    _POSITIVE_WORDS = {"bene", "buono", "great", "good", "ottimo", "perfetto", "nice"}
    _AGREEMENT_WORDS = {"si", "sì", "certo", "ok", "okay", "agree"}
    _CONCLUSION_WORDS = {"conclusione", "quindi", "infine", "finally", "therefore", "alla fine"}

    def __init__(self, registry: FaceMotionRegistry | None = None) -> None:
        self.registry = registry or FaceMotionRegistry()

    def plan(self, text: str, avatar_id: str, language: str) -> List[FaceMotionIntent]:
        words = re.findall(r"\w+|\?", text.lower(), flags=re.UNICODE)
        intents: list[FaceMotionIntent] = []
        used_words: set[str] = set()

        def add_if_missing(motion_id: str, anchor_word: str, text_span: str, intensity: float) -> None:
            if anchor_word in used_words:
                return
            if motion_id not in self.registry.motions:
                return
            used_words.add(anchor_word)
            intents.append(
                FaceMotionIntent(
                    text_span=text_span,
                    motion_id=motion_id,
                    anchor_word=anchor_word,
                    intensity=intensity,
                )
            )

        for word in words:
            if word in self._QUESTION_WORDS:
                add_if_missing("question_face", word, word, 0.7)
                break

        for word in words:
            if word in self._EMPHASIS_WORDS:
                add_if_missing("brow_raise_small", word, word, 0.75)
                break

        for word in words:
            if word in self._POSITIVE_WORDS:
                add_if_missing("smile_small", word, word, 0.7)
                break

        for word in words:
            if word in self._AGREEMENT_WORDS:
                add_if_missing("nod_small", word, word, 0.55)
                break

        for word in words:
            if word in self._CONCLUSION_WORDS:
                add_if_missing("nod_small", word, word, 0.65)
                break

        if not intents:
            add_if_missing("face_idle_soft", words[0] if words else "neutral", text[:24], 0.2)

        return intents[:4]
