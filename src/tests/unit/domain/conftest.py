from typing import Any

import pytest

from deps_ai_fusion.domain.model import (
    ContextAttachments,
    RawLLMExtractionParams,
    RawPageSpan,
)


@pytest.fixture
def raw_extraction_params_factory():
    def _build(
        *,
        custom_instruction: str,
        grouping_factor: int,
        temperature: float,
        top_p: float,
        page_span: RawPageSpan | None,
        context_attachments: ContextAttachments | None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        logprobs: bool | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        llm_params: dict[str, Any] = {}
        if max_tokens is not None:
            llm_params["max_tokens"] = max_tokens
        if stop is not None:
            llm_params["stop"] = stop
        if seed is not None:
            llm_params["seed"] = seed
        if logprobs is not None:
            llm_params["logprobs"] = logprobs
        if model_kwargs is not None:
            llm_params["model_kwargs"] = model_kwargs
        return RawLLMExtractionParams(
            custom_instruction=custom_instruction,
            grouping_factor=grouping_factor,
            temperature=temperature,
            top_p=top_p,
            page_span=page_span,
            context_attachments=context_attachments,
            extra_llm_params=llm_params or None,
        )

    return _build
