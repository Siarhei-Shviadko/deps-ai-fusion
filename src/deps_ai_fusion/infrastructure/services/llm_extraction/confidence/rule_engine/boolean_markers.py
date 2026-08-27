import re
from typing import ClassVar

__all__ = ["BooleanMarkers"]


class BooleanMarkers:
    _TRUE_SYMBOLS: ClassVar[tuple[str, ...]] = (
        "☑",
        "✔",
        "✓",
        "[x]",
        "[X]",
        "(x)",
        "(X)",
        "|x|",
        "|X|",
        ":selected:",
        ":checked:",
    )
    _FALSE_SYMBOLS: ClassVar[tuple[str, ...]] = (
        "☐",
        "□",
        "✗",
        "✘",
        "[ ]",
        "()",
        "( )",
        "|:|",
        "|_|",
        ":unselected:",
        ":unchecked:",
    )

    _TRUE_WORDS: ClassVar[tuple[str, ...]] = (
        "yes",
        "true",
        "checked",
        "selected",
        "enabled",
        "paid",
        "approved",
        "accepted",
        "confirmed",
        "completed",
        "active",
        "valid",
        "signed",
    )
    _FALSE_WORDS: ClassVar[tuple[str, ...]] = (
        "no",
        "false",
        "unchecked",
        "unselected",
        "disabled",
        "unpaid",
        "declined",
        "rejected",
        "denied",
        "pending",
        "inactive",
        "invalid",
        "unsigned",
    )
    _FALSE_PHRASES: ClassVar[tuple[str, ...]] = ("not accepted", "not applicable")

    _TRUE_WORD_RE: ClassVar[re.Pattern] = re.compile("|".join(rf"\b{w}\b" for w in _TRUE_WORDS))
    _FALSE_WORD_RE: ClassVar[re.Pattern] = re.compile("|".join(rf"\b{w}\b" for w in _FALSE_WORDS))
    _FALSE_PHRASE_RE: ClassVar[re.Pattern] = re.compile("|".join(re.escape(p) for p in _FALSE_PHRASES))

    @classmethod
    def find_true_markers(cls, text: str) -> list[str]:
        lower = text.lower()
        symbols = [s for s in cls._TRUE_SYMBOLS if s in lower]
        words = cls._TRUE_WORD_RE.findall(lower)
        return symbols + words

    @classmethod
    def find_false_markers(cls, text: str) -> list[str]:
        lower = text.lower()
        symbols = [s for s in cls._FALSE_SYMBOLS if s in lower]
        words = cls._FALSE_WORD_RE.findall(lower)
        phrases = cls._FALSE_PHRASE_RE.findall(lower)
        return symbols + words + phrases

    @classmethod
    def find_all_markers(cls, text: str) -> list[str]:
        return cls.find_true_markers(text) + cls.find_false_markers(text)
