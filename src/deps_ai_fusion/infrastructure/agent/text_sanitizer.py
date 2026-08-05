__all__ = ["sanitize_text"]


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")
