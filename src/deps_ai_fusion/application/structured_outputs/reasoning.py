from pydantic import BaseModel, Field

__all__ = ["ReasoningResponse"]


class ReasoningResponse(BaseModel):
    reasoning: str = Field(
        ...,
        title="reasoning",
        description="Detailed, step-by-step reasoning process leading to the final response.",
    )
    final_response: str = Field(
        ...,
        title="final_response",
        description="The final answer or result derived from the reasoning process.",
    )
