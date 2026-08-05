from deps_gen_ai.providers import ProviderCode

from deps_ai_fusion.infrastructure.agent.agentic_workflow_factory.provider_factory import (
    ModelProviderFactory,
)
from deps_ai_fusion.infrastructure.agent.settings import AgentSettings


def test_litellm_llm__with_default_model__ok() -> None:
    settings = AgentSettings(
        LITELLM_ENABLED=True,
        LITELLM_BASE_URL="http://localhost:4000",
        LITELLM_API_KEY="sk-test",
        AGENT_MODEL_ID="gpt-4o",
        EPAM_DIAL_ENDPOINT="http://dial",
        EPAM_DIAL_API_KEY="key",
        EPAM_DIAL_API_VERSION="2023-03-15-preview",
    )
    factory = ModelProviderFactory(settings)

    llm = factory.create_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-4o"
    assert llm.openai_api_base == "http://localhost:4000"
    assert llm.openai_api_key.get_secret_value() == "sk-test"
    assert "api-version" in llm.default_headers
    assert llm.max_retries == 3


def test_create_llm__use_litellm_true__returns_chat_openai__ok() -> None:
    settings = AgentSettings(
        LITELLM_ENABLED=True,
        LITELLM_BASE_URL="http://localhost:4000",
        LITELLM_API_KEY="sk-test",
        AGENT_MODEL_ID="gpt-4o",
        EPAM_DIAL_ENDPOINT="http://dial",
        EPAM_DIAL_API_KEY="key",
        EPAM_DIAL_API_VERSION="2023-03-15-preview",
    )
    factory = ModelProviderFactory(settings)

    llm = factory.create_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"


def test_create_llm__use_litellm_false__returns_native_provider__ok() -> None:
    settings = AgentSettings(
        LITELLM_ENABLED=False,
        AGENT_PROVIDER_ID=ProviderCode.OPENAI,
        OPENAI_API_KEY="sk-test",
        AGENT_MODEL_ID="gpt-4o",
        EPAM_DIAL_ENDPOINT="http://dial",
        EPAM_DIAL_API_KEY="key",
        EPAM_DIAL_API_VERSION="2023-03-15-preview",
    )
    factory = ModelProviderFactory(settings)

    llm = factory.create_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
