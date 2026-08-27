import ast
import json
from typing import Any, Callable

from deps_gen_ai.common import LLMResponse

__all__ = ["InsightParser"]

_FINAL_RESPONSE_KEYS = ("final_response", "Final Response")


class InsightParser:
    def extract_content(self, response: LLMResponse[Any]) -> str:
        if response.parsed and hasattr(response.parsed, "final_response"):
            return str(response.parsed.final_response)
        if response.parsed and isinstance(response.parsed, dict):
            for key in _FINAL_RESPONSE_KEYS:
                value = response.parsed.get(key)
                if value is not None:
                    return value
            return json.dumps(response.parsed, indent=2, ensure_ascii=False)
        return self._extract_leading_json(response.content)

    def parse_item(self, content: str) -> dict | None:
        parsed = self._try_load(content, json.loads) or self._try_load(content, ast.literal_eval)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
        return None

    def content_from(self, item: dict | None, raw_content: str) -> str:
        if item is None:
            return raw_content
        if isinstance(item.get("values"), list):
            return raw_content
        return str(item.get("value", raw_content))

    def _try_load(self, content: str, loader: Callable) -> Any:
        try:
            return loader(content)
        except Exception:
            return None

    def _extract_leading_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("{"):
            end = self._find_json_end(text, "{", "}")
        elif text.startswith("["):
            end = self._find_json_end(text, "[", "]")
        else:
            return text
        return text[: end + 1] if end != -1 else text

    def _find_json_end(self, text: str, opener: str, closer: str) -> int:
        depth = 0
        for i, ch in enumerate(text):
            depth += (ch == opener) - (ch == closer)
            if not depth:
                return i
        return -1
