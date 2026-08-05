# Changelog: Agent Instructions (`system_message.py`)

## Context

`system_message.py` contains system messages for two agent modes:
- `SYSTEM_MESSAGE_EXISTING_DOCUMENT_TYPE` — agent knows the document type, creates GenAI fields
- `SYSTEM_MESSAGE_UNKNOWN_DOCUMENT_TYPE` — agent creates a document type from scratch (bootstrap)

Both messages are assembled from shared blocks: `_PROMPTS_CHAIN_FORMAT` + `_SHARED_PRINCIPLES`.

---

## Changes

### 1. Prompt template (`<prompt_template>`)

**Before:** no explicit template — the agent could output prompts in any format it chose.

**After:** a `<prompt_template>` block was added with a full structure the agent must follow.

#### When to use the full template vs. a short prompt

```
- KEY_VALUE_PAIR (SCALAR or LIST): always use the full template
- STRING / BOOLEAN with non-trivial extraction logic: use the full template
- STRING / BOOLEAN with simple, unambiguous extraction: use a concise single-sentence prompt
```

#### Template structure

```
## Objective
You are an information extraction model.
Extract the field(s) "<FIELD_NAME>" located in the "<SECTION_NAME>" of the document.
[OPTIONAL: brief business context]
[OPTIONAL: section-tracking block — only when user explicitly requests aliased extraction]
**IMPORTANT RULE: [one of: "Return ONLY ONE entry per key." / "You may return multiple entries..."]**
Return <KEY-VALUE ENTRY NAMES> in the key_value_pairs list.

## Field Instructions
### <Entry Name>
**Context:** ...
**Field Labels:** ...
**What to capture:**
  - key → ...
  - value → ...
  - alias → ... (only if section-tracking was requested)
**Extraction Instructions:**
1) ...
[OPTIONAL: **Note:** — only for genuine edge cases]

## Disambiguation Rules
- ...

## Normalization
- ...

## Transformation Rules   ← OPTIONAL
[only if user explicitly requested transformation]
```

**Mandatory sections:** Objective, Field Instructions, Disambiguation Rules, Normalization.  
**Optional:** business context sentence, alias line, section-tracking block, Note lines, Transformation Rules.

#### Placeholder rule

An explicit requirement was added: the agent must replace every `<...>` and `[...]` with real content. Placeholder instructions must never appear in the final prompt.

---

### 2. Objective wording

**Before:**
```
You are an information extraction model for re/insurance contracts.
Find the fields...
```

**After:**
```
You are an information extraction model.
Extract the field(s) "<FIELD_NAME>" located in the "<SECTION_NAME>" of the document.
[OPTIONAL: brief business context...]
```

Changes:
- Removed re/insurance specialization (the agent works with any document type)
- `Find` → `Extract` (more precise verb)
- Field name and section are now explicitly named in the Objective
- Business context moved to an optional line

---

### 3. Disambiguation Rules

Expanded from a single line to an explicit prioritized list:

```
- Extract values closest to the field label(s)
- Prefer structured tables or clearly labeled values over narrative text
- If multiple values found, choose by:
  - proximity to field label
  - structured/formatted location (e.g., table, form field)
  - completeness of information
  - order in document
  - confidence
- Return [ONLY ONE / EACH DISTINCT] entry per [key / key per section]
- If no field found, provide an empty value
```

---

### 4. Transformation Rules — new optional section

**Before:** no transformation section existed.

**After:** a `## Transformation Rules` section was added to the template, marked as OPTIONAL.

Inclusion logic:
- The agent does **not** include this section by default
- When proposing a prompt, the agent **must** ask the user whether transformation is needed
- The section is included only if the user explicitly describes the rules (e.g., "dates as YYYY-MM-DD")

---

### 5. Behavior before calling `perform-llm-extraction`

**Before:** the agent could call the test tool without showing the prompt to the user first.

**After:** a hard requirement was added to `_SHARED_PRINCIPLES`:

> When presenting proposed prompt(s) for user review, the response **must** end with:
> "Do you need any data transformation applied (e.g., date formatting, unit conversion, code mapping), or shall I proceed with a test extraction as-is?"
>
> How to interpret the user's reply:
> - "no" / "proceed" / "yes, test it" → no transformation needed, call `perform-llm-extraction`
> - A transformation description (e.g., "dates as YYYY-MM-DD") → add a Transformation Rules section to the prompt, show the updated prompt, ask again
>
> `perform-llm-extraction` must not be called until this question has been answered.

Both system message `tool_guidance` blocks were updated to reflect this: `perform-llm-extraction` may only be called after the user has answered the transformation question.

---

### 6. Examples — expanded from one to three

**Before:** one example (simple STRING field `invoice_total_amount`) with no template body, plus `approve-and-create`. No transformation question.

**After:** three examples with explicit IDs:

| ID | Scenario |
|---|---|
| `simple-string` | STRING field → short single-sentence prompt |
| `complex-kvp` | KEY_VALUE_PAIR → full template with all sections (Deductible example) |
| `approve-and-create` | User confirms → agent calls `create-genai-field` |

All examples now end with the transformation question. The `complex-kvp` example demonstrates the full template structure — per Anthropic best practice, examples are the most reliable format-steering mechanism.

---

## Related file changes

| File | Change |
|---|---|
| `system_message.py` | Main changes described above |
| `tools/list_document_type_fields.py` | New tool — lists existing GenAI Fields; referenced in `genai_field_creation.py` when-NOT-to-use |
| `tools/genai_field_creation.py` | Updated when-NOT-to-use: must call `list-document-type-fields` first |
