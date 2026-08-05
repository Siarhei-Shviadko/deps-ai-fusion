from typing import Any

from deps_gen_ai.providers import ProviderCode
from pydantic import Field

from deps_ai_fusion.domain.model import RawLLMExtractor

from ..base import ConfiguredBaseModel
from .llm_extraction_params import SerializedLLMExtractionParams

__all__ = ["CreateExtractorRequest", "CreateExtractorResponse"]


class CreateExtractorRequest(ConfiguredBaseModel):
    extractor_name: str = Field(..., alias="extractorName")
    provider: ProviderCode
    model: str
    document_type_name: str = Field(..., alias="documentTypeName")
    extractor_id: str | None = Field(default=None, alias="extractorId")

    extraction_params: SerializedLLMExtractionParams = Field(..., alias="extractionParams")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type_name": self.document_type_name,
            "extractor": RawLLMExtractor(
                extractor_name=self.extractor_name,
                provider=self.provider,
                model=self.model,
                extractor_id=self.extractor_id,
                extraction_params=self.extraction_params.to_dict(),
            ),
        }


class CreateExtractorResponse(ConfiguredBaseModel):
    extractor_id: str = Field(..., alias="extractorId")
    document_type_id: str = Field(..., alias="documentTypeId")
