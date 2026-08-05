import json

import pytest
from pydantic import ValidationError

from deps_ai_fusion.infrastructure.agent.agentic_workflow_factory.workflow_factory import (
    AgenticWorkflowFactory,
)
from deps_ai_fusion.infrastructure.agent.persistance.models import InsightsPayload
from deps_ai_fusion.infrastructure.agent.state import AgentState
from tests.factories import AgentStateFactory
from tests.fakes.insights_store import InMemoryInsightsStore


@pytest.fixture
def agentic_workflow_factory(mocker, fake_insights_store: InMemoryInsightsStore):
    mocker.patch(
        "deps_ai_fusion.infrastructure.agent.agentic_workflow_factory.workflow_factory.AgentSettings",
        return_value=mocker.Mock(),
    )
    mock_llm = mocker.Mock()
    mock_provider = mocker.Mock()
    mock_provider.create_llm.return_value = mock_llm
    mocker.patch(
        "deps_ai_fusion.infrastructure.agent.agentic_workflow_factory.workflow_factory.ModelProviderFactory.from_settings",
        return_value=mock_provider,
    )
    factory = AgenticWorkflowFactory(
        document_loader=mocker.Mock(),
        document_type_creation=mocker.Mock(),
        genai_field_creation=mocker.Mock(),
        list_document_type_fields=mocker.Mock(),
        perform_llm_extraction=mocker.Mock(),
        insights_store=fake_insights_store,
    )
    factory.llm = mock_llm
    return factory


def test_insights_payload_from_assistant_content__plain_json_string__ok() -> None:
    content = '{"insights": ["insight one", "insight two"]}'

    result = AgenticWorkflowFactory._insights_payload_from_assistant_content(content)

    assert isinstance(result, InsightsPayload)
    assert len(result.insights) == 2
    assert result.insights[0] == "insight one"
    assert result.insights[1] == "insight two"


def test_insights_payload_from_assistant_content__markdown_json_fence__ok() -> None:
    content = '```json\n{"insights": ["fenced insight"]}\n```'

    result = AgenticWorkflowFactory._insights_payload_from_assistant_content(content)

    assert len(result.insights) == 1
    assert result.insights[0] == "fenced insight"


def test_insights_payload_from_assistant_content__list_of_text_parts__ok() -> None:
    content = [
        {"type": "text", "text": '{"insights": ['},
        {"type": "text", "text": '"part insight"]}'},
    ]

    result = AgenticWorkflowFactory._insights_payload_from_assistant_content(content)

    assert len(result.insights) == 1
    assert result.insights[0] == "part insight"


def test_insights_payload_from_assistant_content__invalid_json__error() -> None:
    with pytest.raises(json.JSONDecodeError):
        AgenticWorkflowFactory._insights_payload_from_assistant_content("not json")


def test_insights_payload_from_assistant_content__invalid_schema__error() -> None:
    with pytest.raises(ValidationError):
        AgenticWorkflowFactory._insights_payload_from_assistant_content('{"insights": "not a list"}')


def test_persist_insights__valid_llm_json__merges_and_saves__ok(
    agentic_workflow_factory: AgenticWorkflowFactory,
    agent_state_factory: AgentStateFactory,
    fake_insights_store: InMemoryInsightsStore,
    mocker,
) -> None:
    state: AgentState = agent_state_factory(insights=["existing insight"])
    reply = mocker.Mock()
    reply.content = '{"insights": ["new insight"]}'
    agentic_workflow_factory.llm.invoke.return_value = reply

    result = agentic_workflow_factory._persist_insights(state)

    assert len(result.insights) == 2
    assert set(result.insights) == {"existing insight", "new insight"}
    stored = fake_insights_store.load(state.conversation_id, state.tenant_id)
    assert len(stored) == 2
    assert set(stored) == {"existing insight", "new insight"}


def test_persist_insights__invalid_llm_response__state_unchanged__ok(
    agentic_workflow_factory: AgenticWorkflowFactory,
    agent_state_factory: AgentStateFactory,
    fake_insights_store: InMemoryInsightsStore,
    mocker,
) -> None:
    state: AgentState = agent_state_factory(insights=["keep me"])
    original_insights = list(state.insights)
    reply = mocker.Mock()
    reply.content = "not valid json"
    agentic_workflow_factory.llm.invoke.return_value = reply

    result = agentic_workflow_factory._persist_insights(state)

    assert result.insights == original_insights
    assert fake_insights_store.load(state.conversation_id, state.tenant_id) == []


def test_persist_insights__empty_insights_and_no_existing__state_unchanged__ok(
    agentic_workflow_factory: AgenticWorkflowFactory,
    agent_state_factory: AgentStateFactory,
    fake_insights_store: InMemoryInsightsStore,
    mocker,
) -> None:
    state: AgentState = agent_state_factory(insights=[])
    reply = mocker.Mock()
    reply.content = '{"insights": []}'
    agentic_workflow_factory.llm.invoke.return_value = reply

    result = agentic_workflow_factory._persist_insights(state)

    assert result.insights == []
    assert fake_insights_store.load(state.conversation_id, state.tenant_id) == []
