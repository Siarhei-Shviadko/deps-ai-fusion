# GenAI Queries Agent

## Graph

```mermaid
flowchart TD
    START([__start__]) --> load_insights
    load_insights --> assemble_context

    assemble_context -->|document_type_id is set| main_agent
    assemble_context -->|document_type_id is None| bootstrap_agent

    bootstrap_agent -->|document_type_id is set| main_agent
    bootstrap_agent -->|document_type_id still None| persist_insights

    main_agent --> persist_insights
    persist_insights --> END([__end__])

    subgraph bootstrap_agent [bootstrap_agent — ReAct]
        B_LLM[LLM] -->|tool call| B_TOOLS[tools:\nload-document\nperform-llm-extraction\ncreate-document-type]
        B_TOOLS -->|result| B_LLM
    end

    subgraph main_agent [main_agent — ReAct]
        M_LLM[LLM] -->|tool call| M_TOOLS[tools:\nload-document\nlist-document-type-fields\ncreate-genai-field\nperform-llm-extraction]
        M_TOOLS -->|result| M_LLM
    end
```

### Nodes

| Node | Description |
|------|-------------|
| `load_insights` | Loads persisted insights from previous conversation turns |
| `assemble_context` | Injects insights as a `SystemMessage` into the message history |
| `bootstrap_agent` | ReAct agent used when no Document Type exists yet — creates it |
| `main_agent` | ReAct agent used when a Document Type already exists — manages fields and extraction |
| `persist_insights` | Summarizes the conversation and saves key insights for future turns |

### Routing

- **`assemble_context` → `bootstrap_agent`** — `document_type_id` is not set
- **`assemble_context` → `main_agent`** — `document_type_id` is already known
- **`bootstrap_agent` → `main_agent`** — agent successfully created a Document Type during the turn
- **`bootstrap_agent` → `persist_insights`** — agent could not create a Document Type (skips main agent)
