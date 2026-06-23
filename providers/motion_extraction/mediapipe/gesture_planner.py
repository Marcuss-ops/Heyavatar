from __future__ import annotations

import re
from typing import List

from contracts.gesture_planner import GesturePlanner, GestureIntent


class RuleBasedGesturePlanner(GesturePlanner):
    """Keyword planner for co-speech gestures.

    The goal is not perfect NLP. The goal is to reliably schedule a
    handful of gesture families so the avatar looks intentional:
    counts, emphasis, comparison, question, and conclusion.
    """

    _COUNT_WORDS = {
        "uno": "count_one",
        "one": "count_one",
        "due": "count_two",
        "two": "count_two",
        "tre": "count_three",
        "three": "count_three",
    }
    _EMPHASIS_WORDS = {"molto", "importante", "fondamentale", "chiave", "really", "important"}
    _QUESTION_WORDS = {"?", "perché", "perche", "why", "what", "come", "how"}
    _CONCLUSION_WORDS = {"conclusione", "quindi", "infine", "finally", "therefore", "alla fine"}
    _COMPARISON_WORDS = {"vs", "contro", "between", "differenza", "difference", "compare"}
    _OPEN_WORDS = {"ciao", "benvenuti", "welcome", "oggi", "today"}

    def plan(self, text: str, avatar_id: str, language: str) -> List[GestureIntent]:
        words = re.findall(r"\w+|\?", text.lower(), flags=re.UNICODE)
        intents: list[GestureIntent] = []
        used_words: set[str] = set()

        def add_if_missing(gesture_id: str, anchor_word: str, text_span: str, intensity: float) -> None:
            if anchor_word in used_words:
                return
            used_words.add(anchor_word)
            intents.append(
                GestureIntent(
                    text_span=text_span,
                    gesture_id=gesture_id,
                    anchor_word=anchor_word,
                    intensity=intensity,
                )
            )

        for word in words:
            if word in self._COUNT_WORDS:
                add_if_missing(self._COUNT_WORDS[word], word, word, 0.85)
                break

        for word in words:
            if word in self._QUESTION_WORDS:
                add_if_missing("question", word, word, 0.75)
                break

        for word in words:
            if word in self._COMPARISON_WORDS:
                add_if_missing("comparison", word, word, 0.8)
                break

        for word in words:
            if word in self._CONCLUSION_WORDS:
                add_if_missing("conclusion", word, word, 0.7)
                break

        for word in words:
            if word in self._EMPHASIS_WORDS:
                add_if_missing("emphasis_small", word, word, 0.9)
                break

        for word in words:
            if word in self._OPEN_WORDS:
                add_if_missing("open_palms", word, word, 0.55)
                break

        if not intents:
            add_if_missing("idle_small", words[0] if words else "neutral", text[:24], 0.2)

        return intents[:3]
