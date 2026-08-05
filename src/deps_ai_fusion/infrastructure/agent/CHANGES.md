# Changes: Agent Prompt & Tool Description Improvements

## Context

The changes below apply Anthropic 2025 best practices for LLM system prompts and tool-use descriptions to the GenAI Queries Agent.

---

## 1. `agentic_workflow_factory/system_message.py`

### 1.1 XML tags instead of plain-text section headers

**Before:**
```
Goals:
- ...
Operating principles:
- ...
Output style:
- ...
```

**After:**
```xml
<role>...</role>
<tool_guidance>...</tool_guidance>
<reasoning_instructions>...</reasoning_instructions>
<constraints>...</constraints>
<output_format>...</output_format>
```

**Why:** Claude is trained to parse XML tags reliably. Plain-text headers (`Goals:`, `Operating principles:`) are more ambiguous and offer no structural guarantees.

---

### 1.2 Shared principles extracted into `_SHARED_PRINCIPLES` and `_PROMPTS_CHAIN_FORMAT` constants

**Before:** `<reasoning_instructions>`, `<constraints>`, `<output_format>`, and prompts_chain guidance were duplicated in both `SYSTEM_MESSAGE_UNKNOWN_DOCUMENT_TYPE` and `SYSTEM_MESSAGE_EXISTING_DOCUMENT_TYPE` with slightly different wording.

**After:** Common content lives in two module-level constants interpolated via f-strings into both messages.

**Why:** Single source of truth — a rule change now requires editing one place instead of two. The original duplication had already led to three pairs of divergent formulations for the same rules.

---

### 1.3 `Assumptions` section removed

**Before:**
```
Assumptions:
- You are in bootstrap mode because the current conversation state does not include a Document Type.
```

**After:** Section removed entirely.

**Why:** This restates information the LLM already has from the graph state. It consumes tokens without adding value.

---

### 1.4 Post-tool reasoning instruction added

**Before:** No explicit guidance on what to do after a tool returns a result.

**After:**
```
After receiving a tool result: reflect on its quality before deciding the next step.
Do not repeat the same tool call unless state has changed.
```

**Why:** Anthropic 2025 best practice for agentic loops — explicit post-result reflection reduces redundant tool calls and improves decision quality.

---

### 1.5 Parallel tool call instruction added

**Before:** No guidance on parallel tool execution.

**After:**
```
Make independent tool calls in parallel when results do not depend on each other.
```

**Why:** Claude 4.x natively supports parallel tool calls. Without explicit guidance, the model defaults to sequential calls even when parallelism is safe.

---

### 1.6 "When NOT to use" added to `<tool_guidance>` for each tool

**Before:** Tool guidance described only when to use a tool.

**After:** Each tool entry now includes an explicit `Do not use if...` clause.

**Why:** Anthropic documentation states this is the single most important factor in tool selection quality.

---

### 1.7 `<prompts_chain_format>` block added

**Before:** No description of what a `prompts_chain` is structurally.

**After:**
```xml
<prompts_chain_format>
A prompts_chain is a list of prompt strings executed sequentially...
</prompts_chain_format>
```

**Why:** Without a structural definition the agent may infer an incorrect schema from context instead of following the one defined in `schemas.py`.

---

### 1.8 `<data_types>` block added

**Before:** The agent had no explicit list of valid `data_type` / `cardinality` combinations.

**After:**
```xml
<data_types>
- STRING / SCALAR, STRING / LIST
- BOOLEAN / SCALAR, BOOLEAN / LIST
- KEY_VALUE_PAIR / SCALAR, KEY_VALUE_PAIR / LIST (supports include_aliases)
</data_types>
```

**Why:** Without this, the agent must infer valid types from the tool schema at call time, increasing the chance of hallucinated or invalid type combinations.

---

### 1.9 `<examples>` block added

**Before:** No examples of desired agent behaviour.

**After:** Two concrete interaction examples — proposing a field with confirmation, and creating it after user approval.

**Why:** Anthropic calls `<examples>` "the most reliable format-steering mechanism." Even one good example reduces format deviation significantly.

---

### 1.10 `<bootstrap_completion>` block preserved and scoped

The instruction to provide a summary after Document Type creation was retained in `SYSTEM_MESSAGE_UNKNOWN_DOCUMENT_TYPE` only (it does not apply to the main agent).

---

## 2. `tools/load_document.py`

### 2.1 Removed duplicated reasoning-length instruction

**Before:** `"Keep reasoning short (<=20 words) explaining why loading is necessary now."`

**After:** Removed.

**Why:** This rule is now centrally defined in `<reasoning_instructions>` in the system message. Repeating it in the tool description creates a second source of truth that can drift.

### 2.2 Added "when NOT to use"

**Added:** `"Do not call if the document layout is already present in context."`

---

## 3. `tools/perform_llm_extraction.py`

### 3.1 Removed duplicated prompts_chain brevity instruction

**Before:** `"Prefer the smallest viable chain (often one prompt); add steps only if necessary."`

**After:** Removed.

**Why:** Centrally defined in `<prompts_chain_format>` in the system message.

### 3.2 Removed duplicated reasoning-length instruction

**Before:** `"Reasoning must state the hypothesis being tested."`

**After:** Removed.

**Why:** Centrally defined in `<reasoning_instructions>` in the system message.

### 3.3 Added "when NOT to use"

**Added:** `"Do not use for direct document questions that do not require structured extraction."`

---

## 4. `tools/document_type_creation.py`

### 4.1 Removed duplicated reasoning-length instruction

**Before:** `"Reasoning must include the user-confirmed name and why creation is needed now."`

**After:** Removed.

**Why:** Centrally defined in `<reasoning_instructions>` in the system message.

### 4.2 Added "when NOT to use"

**Added:** `"Do not use if a Document Type already exists in the current conversation state."`

---

## 5. `tools/genai_field_creation.py`

### 5.1 Removed duplicated prompts_chain brevity instruction

**Before:** `"Prefer the smallest viable prompts_chain (often one prompt); add steps only when strictly necessary."`

**After:** Removed.

**Why:** Centrally defined in `<prompts_chain_format>` in the system message.

### 5.2 Removed duplicated reasoning-length instruction

**Before:** `"Reasoning should summarize the user approval and why creation is safe now."`

**After:** Removed.

**Why:** Centrally defined in `<reasoning_instructions>` in the system message.

### 5.3 Added "when NOT to use"

**Added:** `"Do not use if a field with the same name or purpose may already exist — flag it and ask the user first."`

---

---

## TODO — What Still Needs to Be Fixed

### T2. `_persist_insights` system message in `workflow_factory.py` (medium priority)

**File:** `agentic_workflow_factory/workflow_factory.py`, method `_persist_insights` (~line 156)

The summarizer prompt is a raw inline string without XML structure. It should receive the same treatment as the agent system messages: XML tags, no duplicated instructions, explicit output format.

Current state:
```python
summarizer_system = SystemMessage(
    content=(
        "You are the Insights Summarizer for an Agent.\n"
        "Context: ...\n"
        "Goal: ...\n\n"
        "Include:\n- ...\n\n"
        "Style: ...\n\n"
        "Output format: a JSON object with key 'insights' as an array..."
    )
)
```

Suggested: wrap sections in `<role>`, `<goal>`, `<include>`, `<style>`, `<output_format>` tags and extract to a named constant outside the method.

---

### T3. `schemas.py` — sparse field descriptions (low priority)

**File:** `tools/schemas.py`

Several `Field(description=...)` values are minimal and do not give the LLM enough context:

| Field | Current description | Problem |
|-------|-------------------|---------|
| `DataShape.data_type` | `"Atomic type of the extracted value(s). Examples: STRING, BOOLEAN, KEY_VALUE_PAIR."` | Lists examples instead of the full valid set; agent may try other values |
| `DocumentTypeCreationRequest.reasoning` | `"Reason for creating the document type."` | No length constraint, no mention that it must reference user approval |
| `ExecuteLLMExtractionRequest.prompts_chain` | `"Ordered prompts to run for this test."` | Much sparser than `CreateGenAIFieldRequest.prompts_chain` which explains the chain structure |

Suggested fix for `DataShape.data_type`:
```python
description="Extraction value type. Must be one of: STRING, BOOLEAN, KEY_VALUE_PAIR."
```

---

### T4. `perform_llm_extraction.py` — `name="It doesn't matter"` code smell (low priority)

**File:** `tools/perform_llm_extraction.py`, method `_resolve_extractor` (~line 75)

When no real extractor exists, a temporary one is created with `name="It doesn't matter"`. This is ambiguous and would surface in any log or error message containing the extractor name. Replace with a descriptive constant:

```python
_TEMP_EXTRACTOR_NAME = "temporary-validation-extractor"
```

---

### T5. `available_tools()` parameters are hardcoded and incorrect (medium priority)

**File:** `agentic_workflow_factory/workflow_factory.py`, method `available_tools` (~line 70)

All tools are registered with `parameters=[{"name": "document_id"}]` regardless of their actual schema. `create-genai-field` gets an extra `document_type_id` entry, but the real schemas (`CreateGenAIFieldRequest`, `ExecuteLLMExtractionRequest`, etc.) have many more fields. This manifest is sent to the external registry and misleads callers.

Suggested fix: derive parameters from the tool's `args_schema` at runtime instead of hardcoding them.

---

### T6. `<tool_guidance>` in system message partially duplicates tool descriptions (low priority)

**File:** `agentic_workflow_factory/system_message_new.py`

After adding "when NOT to use" clauses to the tool `description` fields (changes 2–5 above), the `<tool_guidance>` block in the system message now overlaps with those descriptions. Both sources currently give guidance on when to call and when not to call each tool.

Two options:
- **Remove `<tool_guidance>` from system message** — rely solely on tool descriptions, which are passed to the LLM as part of the tool schema. Reduces token count and eliminates the second maintenance point.
- **Keep both but make roles explicit** — system message `<tool_guidance>` covers behavioral/sequencing rules; tool `description` covers schema-level constraints. Document this separation with a comment.

---

## Summary table

| File | Change type | Items |
|------|-------------|-------|
| `system_message.py` | Modified | XML tags, shared constants, removed Assumptions, post-tool reasoning, parallel calls, when-NOT-to-use in tool_guidance, prompts_chain_format, data_types, examples, bootstrap_completion |
| `load_document.py` | Modified | Removed duplicated reasoning instruction, added when-NOT-to-use |
| `perform_llm_extraction.py` | Modified | Removed 2 duplicated instructions, added when-NOT-to-use |
| `document_type_creation.py` | Modified | Removed duplicated reasoning instruction, added when-NOT-to-use |
| `genai_field_creation.py` | Modified | Removed 2 duplicated instructions, added when-NOT-to-use |
| `tools/list_document_type_fields.py` | New file | New tool — lists existing GenAI Fields for a Document Type and extractor |