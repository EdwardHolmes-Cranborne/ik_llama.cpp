# WIR State Batch Compiler

## Multi-State Document Compilation for Parallel Prefill Refinement

**Status:** Specification v0.2 — DRAFT
**Author:** Ed Holmes (Cranborne Audio)
**Date:** March 2026
**Depends on:** WIR State Refinement Method v0.1

---

## 1. Abstract

This document specifies a toolchain for compiling multiple independent state documents into a single structured file that can be submitted to a WIR-enabled inference engine as a batch of parallel refinement prompts. The compiler accepts heterogeneous input formats — spreadsheets, markdown documents, parameter range definitions, and prompt templates — and produces a unified state batch file containing one state document per logical entity.

The compiler may invoke an LLM (via an LM Studio OpenAI-compatible endpoint) during compilation to expand terse parameter definitions into well-formed state documents, generate initial prose sections from structured data, and validate that compiled state documents are syntactically and semantically well-formed.

The output is a single `.wirstate` file containing N state documents with metadata headers, suitable for parallel processing by llama-wir or llama-server with batch scheduling.

---

## 1.1 Two-Phase Architecture

WIR state processing operates in two distinct phases:

**Phase A: Refinement** — 1000 independent micro-prompts, batched in a single weight-streaming pass.
**Phase B: Generation** — auto-translated readable markdown table injected into the model's generation context.

### 1.1.1 Four-Zone Micro-Prompt Architecture (Phase A: Refinement)

During refinement, each variable is refined as its own tiny conversation — a **micro-prompt**. The four zones apply **per micro-prompt**, not as one big document. Each micro-prompt has perfect zone ordering:

| Zone | Contents | Changes when | KV cache behaviour | Approx. tokens |
|---|---|---|---|---|
| **Zone 1: INSTRUCTION** | Variable name, type, range, interpretation guide | Never (authored at compile time) | Computed once, cached to disk, reused forever | ~20 tokens |
| **Zone 2: CONTEXT** | Relevant excerpt from the conversation or event stream | New conversation turn or event arrives | Recomputed once per turn, cached across WIR passes within that turn | ~50 tokens |
| **Zone 2b: EXTERNAL STATE** | Hook-injected ground truth: sensor data, API results, system metrics, authoritative values | When external hooks fire (typically per-turn, at pre-refine hook point) | Same caching as Zone 2 — recomputed per-turn, cached across WIR passes | ~10 tokens |
| **Zone 3: ANSWER** | The mutable value: a single byte (single Unicode character), boolean (Y/N), or short string | Every WIR refinement pass | Recomputed each pass (1 token for byte/bool) | 1 token |

Each micro-prompt is ~80 tokens total. Zone ordering within each micro-prompt maximises prefix cache hits: the model reads what the variable means (Zone 1), what just happened (Zone 2), what external systems report (Zone 2b), and only then evaluates its answer (Zone 3).

**Why micro-prompts instead of one big document:** In a single large document with all variables, the attention distance between a variable's instruction (at the top) and its answer (at the bottom) can be hundreds or thousands of tokens. With micro-prompts, each variable's instruction is only ~80 tokens from its answer — perfect attention proximity. All micro-prompts are batched in a single weight-streaming pass, so the cost is the same as processing one large document.

**Per-variable n-least-conf:** Each variable independently decides whether to update. Because each micro-prompt is its own independent context, there are no cross-variable attention distance problems.

**Authority model:** Zone 2b values are authoritative. If a hook sets `cpu_load: 178` and the model's Zone 3 answer says `system_healthy?: Y`, the model sees the contradiction during refinement and corrects Zone 3. Hooks provide data; the model provides interpretation. The inference engine enforces this by marking only Zone 3 tokens as mutable during WIR refinement — Zones 1, 2, and 2b are read-only.

### 1.1.2 Generation Context (Phase B: Readable Table)

After all micro-prompts converge, the orchestrator auto-translates the compact single-char tokens back to human/LLM-readable values and assembles them into a markdown table. This table is injected into the model's generation context — it is what the model reads when generating its actual response.

The translation is a simple lookup — no LLM call needed. The `.wirstate.meta` file or compile-time config defines named value bands for each variable. See section 4.4 for details.

---

## 2. Motivation

### 2.1 The Multi-Agent State Problem

The WIR State Refinement Method (companion spec) describes maintaining a single state document for a single agent or session. Many practical applications require maintaining tens to thousands of state documents simultaneously:

- **Game worlds:** Each NPC has its own state document (beliefs, goals, relationships, inventory). A 200-NPC village is 200 state documents refined in parallel.
- **Fleet management:** Each vehicle, device, or worker has a state document tracking its status, assignment, and history.
- **Simulation:** Each simulated entity maintains state that evolves with each tick.
- **Parameter sweeps:** A researcher tests N configurations of an agent by creating N state documents with different initial parameters and running them in parallel.
- **Multi-persona assistants:** A single system maintains separate state documents for different user relationships, projects, or roles.

Creating these state documents by hand is impractical. The state batch compiler automates the generation of well-formed state documents from compact input descriptions.

### 2.2 Why a Single File

Submitting N separate state files to the inference engine requires N separate API calls, N separate KV cache lookups, and N context switches. A single batch file enables:

- **Batch scheduling:** The inference engine processes all states in a single weight-streaming pass, amortising the cost of loading model weights across all states.
- **Shared prefix caching:** States that share a common system prompt or identity section benefit from a single prefix cache hit rather than N independent lookups.
- **Atomic updates:** All states are refined together in one pass, ensuring temporal consistency (all states see the same "now").
- **Simplified orchestration:** One file in, one file out. No file management for thousands of individual state files.

---

## 3. Input Formats

The compiler accepts four input formats, each suited to different authoring workflows. Inputs can be combined: a spreadsheet defines the entities, a markdown document provides prose templates, and a parameter file defines sweep ranges.

### 3.1 Spreadsheet (CSV/TSV/XLSX)

Each row defines one state document. Columns map to state sections or compact variables.

```csv
entity_id,role,personality,current_goal,confidence_level,task_priority,custom_context
npc_001,blacksmith,gruff but fair,complete the king's order,200,3,"Specialises in ceremonial blades"
npc_002,herbalist,gentle and curious,find moonpetal flowers,180,2,"Recently arrived from the eastern provinces"
npc_003,guard,stern and loyal,patrol the north wall,220,1,"Veteran of the border wars"
```

Column mapping rules:
- `entity_id` (required): Unique identifier. Becomes the state document's ID in the batch file.
- Columns matching known variable names (`confidence_level`, `task_priority`, etc.) are placed in the `[VARIABLES]` section with appropriate encoding (byte-value for integers 0-255, boolean for Y/N values, short string for text).
- Columns matching known section names (`role` -> `[IDENTITY]`, `current_goal` -> `[GOALS]`, `custom_context` -> `[CONTEXT]`) are placed in the corresponding prose section.
- Unrecognised columns are placed in `[CONTEXT]` as key-value pairs.

### 3.2 Markdown Document

A markdown document with repeated section patterns, one per entity. Suitable for hand-authored or LLM-generated state descriptions.

```markdown
# States

## npc_001: Aldric the Blacksmith

### Identity
Gruff but fair master blacksmith. Third generation in the trade.
Takes pride in work but resents the nobility's demands.

### Goals
- Complete the king's ceremonial sword by the harvest festival
- Train apprentice Marten to journeyman level
- Source better iron from the mountain traders

### Context
Specialises in ceremonial blades. Workshop on Market Street.
Currently behind schedule due to supply shortages.

### Variables
confidence_level: 200
task_priority: 3
stressed?: Y
apprentice_progress: 140

---

## npc_002: Elara the Herbalist
...
```

The compiler parses each `## entity_id: Name` block as a separate state document, extracting sections by their `### Section` headers.

### 3.3 Parameter Range File (YAML)

Defines a combinatorial or sampled sweep across parameter values. The compiler generates one state document per parameter combination.

```yaml
# parameter-sweep.yaml
template: "research_agent"
prompt: "You are a research agent investigating {topic}."

parameters:
  topic:
    values: ["climate models", "protein folding", "quantum error correction"]
  search_depth:
    type: byte
    range: [50, 100, 150, 200, 250]
  risk_tolerance:
    type: byte
    range: [30, 128, 220]
  methodology:
    values: ["systematic review", "hypothesis-driven", "exploratory"]

mode: cartesian  # or "sampled" with count
# cartesian: 3 x 5 x 3 x 3 = 135 state documents
# sampled: random N from the cartesian product

identity_template: |
  Research agent specialising in {topic}.
  Methodology: {methodology}.
  Configured for {'conservative' if risk_tolerance < 100 else 'moderate' if risk_tolerance < 200 else 'aggressive'} exploration.
```

For cartesian mode, this generates 135 state documents with all combinations. For sampled mode, a random subset is selected. The `identity_template` is expanded per-combination using Python f-string-style interpolation.

### 3.4 Prompt List (Plain Text or JSON)

A simple list of prompts, each generating one state document. Useful when the state documents are largely identical except for a single varying instruction.

```json
{
  "base_identity": "You are a code review assistant.",
  "base_variables": {
    "strictness": 180,
    "verbosity": 100,
    "focus_security": true
  },
  "prompts": [
    "Review the authentication module for SQL injection vulnerabilities.",
    "Review the payment processing pipeline for PCI compliance.",
    "Review the user input validation layer for XSS vectors.",
    "Review the session management code for CSRF protection."
  ]
}
```

Each prompt becomes a separate state document with the shared identity and variables, plus the prompt placed in the `[GOALS]` section.

---

## 4. Output Format

### 4.1 Micro-Prompt Batch Format (Phase A: Refinement)

The compiled output for refinement uses the micro-prompt architecture described in section 1.1.1. Each variable becomes its own independent micro-prompt with perfect zone ordering. All micro-prompts are batched in a single weight-streaming pass.

For a **single variable micro-prompt**, the layout is:

```
#MICROPROMPT var=user_frustration entity=npc_001
====ZONE1====
user_frustration (byte 0-249, 0=calm 249=furious, single-char token):
Char-to-value: !=0, #=2, $=3, ... (249-char safe table)
====ZONE2====
Previous assistant response: "The bug is on line 38 of auth.py..."
User's latest message: "that didn't work, I told you the error
is on line 42 not line 38"
====ZONE2B====
alerts_active?: Y
errors_last_5m: 042
====ZONE3====
Ñ
```

A **batch of micro-prompts** for all variables of a single entity:

```
#WIRSTATE v1
#ENTITY npc_001
#VARS 7
#FORMAT microprompt_batch
#TOKENS_PER_MICROPROMPT ~80

#MICROPROMPT var=user_frustration
====ZONE1====
user_frustration (byte 0-249, 0=calm 249=furious, single-char token):
====ZONE2====
User's latest message: "that didn't work, I told you the error is on line 42 not line 38"
====ZONE2B====
errors_last_5m: 042
====ZONE3====
Ñ

#MICROPROMPT var=task_completion_pct
====ZONE1====
task_completion_pct (byte 0-249, mapped to 0-100%, single-char token):
====ZONE2====
User's latest message: "that didn't work, I told you the error is on line 42 not line 38"
====ZONE2B====
====ZONE3====
:

#MICROPROMPT var=context_stale
====ZONE1====
context_stale? (Y=needs refresh, N=current):
====ZONE2====
User's latest message: "that didn't work, I told you the error is on line 42 not line 38"
====ZONE2B====
====ZONE3====
Y

... (one micro-prompt per variable)
```

**KV cache behaviour per micro-prompt:**
- Zone 1 (~20 tokens): Variable instruction, identical across all turns and WIR passes. KV computed once, cached to disk, loaded on every call. Cost after first interaction: **zero**.
- Zone 2 (~50 tokens): Conversation context, changes per turn. KV recomputed once per turn, then cached across all WIR passes within that turn.
- Zone 2b (~10 tokens): External hook values, changes when hooks fire. Same caching as Zone 2.
- Zone 3 (1 token): The answer. Only this single token recomputes per WIR pass.

**Why this is better than the single-document approach:**
- Each micro-prompt is ~80 tokens. The attention distance from instruction to answer is always ~80 tokens, regardless of how many variables exist.
- KV cache works perfectly per micro-prompt: Zone 1 cached forever, Zone 2/2b cached per-turn, only the 1-token Zone 3 answer recomputes per WIR pass.
- No cross-variable attention distance problems. With 200 variables in a single document, the last variable's answer would be ~4000 tokens from the first variable's instruction.
- Per-variable n-least-conf: each variable independently decides whether to update.

**Authority chain:** Within each micro-prompt, the model reads top-to-bottom. By Zone 3, it has absorbed the variable definition (Zone 1), the conversation (Zone 2), and external ground truth (Zone 2b). Zone 2b values are authoritative — if they contradict the Zone 3 answer, the model corrects Zone 3. The inference engine enforces this by marking only the Zone 3 token as WIR-mutable.

### 4.2 Multi-Entity Batch Layout

For multiple entities, each entity generates its own set of micro-prompts. All micro-prompts across all entities are batched in a single weight-streaming pass.

If entities share the same variable definitions, the Zone 1 instruction content is identical across their micro-prompts — the KV cache for these Zone 1 segments is computed once and reused across all entities.

```
#WIRSTATE v1
#COUNT 3
#FORMAT microprompt_batch

# Entity npc_001: 4 variables = 4 micro-prompts
#MICROPROMPT var=confidence_level entity=npc_001
====ZONE1====
confidence_level (byte 0-249, 0=none 249=absolute, single-char token):
====ZONE2====
Aldric the blacksmith. Behind schedule on the king's order.
A merchant just arrived offering high-quality iron at double price.
====ZONE2B====
iron_market_price: 200
days_until_festival: 003
====ZONE3====
É

#MICROPROMPT var=task_priority entity=npc_001
====ZONE1====
task_priority (byte 0-249, 0=idle 249=critical, single-char token):
====ZONE2====
Aldric the blacksmith. Behind schedule on the king's order.
====ZONE2B====
kings_messenger_visited?: Y
====ZONE3====
µ

# Entity npc_002: 4 variables = 4 micro-prompts
#MICROPROMPT var=confidence_level entity=npc_002
====ZONE1====
confidence_level (byte 0-249, 0=none 249=absolute, single-char token):
====ZONE2====
Elara the herbalist. Gentle and curious. Recently arrived.
Moonpetal flowers spotted in the eastern meadow by a traveller.
====ZONE2B====
====ZONE3====
µ

... (one micro-prompt per variable per entity)
```

**Batch KV cache behaviour:**
- Shared Zone 1 instructions: If npc_001 and npc_002 share the same `confidence_level` definition, the Zone 1 KV for that instruction is computed once and reused. For 200 entities with 10 shared variables, this saves 199 redundant Zone 1 computations per variable.
- Per-entity Zone 2: Each entity's context is different. KV computed once per entity per turn, shared across all of that entity's micro-prompts (since Zone 2 content is the same for all variables of the same entity).
- Per-variable Zone 3: Only the 1-token answer recomputes per WIR pass.

### 4.3 Encoding: Single-Token Unicode Characters (Safe Table)

Each byte variable is encoded as a single Unicode character guaranteed to tokenise as exactly 1 token. The character pool is drawn from three ranges that are confirmed single-token across Qwen and Mistral tokeniser families:

- **94 ASCII printable** (U+0021 to U+007E)
- **94 Latin-1 Supplement** (U+00A0 to U+00FF)
- **68 Latin Extended-A** (U+0100 to U+0143)

However, **not all 256 are safe for use as value tokens.** Some characters have special meaning to the model or to the wirstate format itself. Using them as values risks the model misinterpreting a value token as a control sequence, closing tag, or template delimiter.

#### Excluded Characters (7 definite risks)

| Char | Code point | Risk |
|---|---|---|
| `"` | U+0022 | String delimiter — conflicts with short string variable quoting |
| `<` | U+003C | Special token start — model may parse `<\|endoftext\|>` |
| `>` | U+003E | Special token end — paired with `<` |
| `\` | U+005C | Escape character — model interprets next character |
| `\|` | U+007C | Chat template pipe — used in `<\|...\|>` |
| NBSP | U+00A0 | Invisible — confuses alignment, model may ignore |
| SHY | U+00AD | Invisible soft hyphen — model may ignore or strip |

#### Borderline Characters (9, included by default)

| Char | Code point | Risk | Why included |
|---|---|---|---|
| `#` | U+0023 | Header/comment marker | Not structural inside Zone 3 values |
| `'` | U+0027 | Apostrophe | Not a wirstate delimiter |
| `/` | U+002F | Closing tags `</s>` | Only dangerous as `</` pair, not standalone |
| `=` | U+003D | Zone markers `====` | Only dangerous as `====` sequence, not standalone |
| `` ` `` | U+0060 | Code fences | Only dangerous as triple ` ``` `, not standalone |
| `[` | U+005B | Section markers | Context makes value role clear |
| `]` | U+005D | Section markers | Paired with `[` |
| `{` | U+007B | Template braces | Not in a template context |
| `}` | U+007D | Template braces | Paired with `{` |

#### Final Safe Table

With 7 exclusions: **249 safe single-token values** (97% of full byte range, ~0.4% granularity). This is the default table.

Conservative mode (also excluding 9 borderline chars): **240 values** (~94% of byte range).

The table is ordered by code point. Value 0 maps to the first safe character, value 248 maps to the last. Values are sequential with no gaps — the lookup table handles the mapping.

**Properties:**
- **Single-token guaranteed:** Each byte variable occupies exactly 1 token during WIR refinement.
- **Fixed-width:** Always 1 character, 1 token. No position shifts across passes.
- **Semantically safe:** No characters that could trigger special model behaviour (EOS, template parsing, escape sequences).
- **Machine-parseable:** Single-character extraction; lookup table maps character to value.

**Lookup table locations:**
1. **Zone 1 instructions** — the model reads the char-to-value table to understand what each character means.
2. **`.wirstate.meta` file** — application code decodes values without parsing Zone 1.

**Model-specific token tables (late goal):** Different model families may have different single-character token sets. The compiler will eventually auto-generate safe lookup tables matched to the loaded model's tokeniser by:
1. Querying the GGUF vocab for all single-character tokens.
2. Filtering out tokens that appear in the model's special token list (EOS, BOS, chat template tokens).
3. Filtering out invisible/control characters.
4. Building the optimal mapping from the remaining safe characters.

For now, the 249-char safe table covering Qwen/Mistral families is hardcoded.

**Booleans** remain `Y`/`N` (1 token). **Short strings** remain quoted, padded to their declared max token width.

### 4.4 Generation Context Format (Phase B: Readable Table)

After all micro-prompts converge during refinement, the orchestrator auto-translates compact single-char tokens back to human/LLM-readable values and assembles a markdown table for the model's generation context.

#### 4.4.1 Auto-Translation

During refinement, the model works with compact single-char tokens:
```
mood: &          (& = value 244)
frustration: Ñ   (Ñ = value 176)
```

For the generation context, the orchestrator auto-translates these to readable form:
```
mood: contentment (244/249)
frustration: elevated (176/249)
```

The translation is a simple lookup — no LLM call needed. Each variable can define named value bands at compile time in the `.wirstate.meta` file or the variable definition:

```yaml
# Variable definition with named bands
user_frustration:
  type: byte
  range: [0, 249]
  bands:
    0-50: "calm"
    51-120: "mild"
    121-180: "elevated"
    181-249: "furious"

mood:
  type: byte
  range: [0, 249]
  bands:
    0-62: "distressed"
    63-124: "neutral"
    125-186: "pleasant"
    187-249: "contentment"
```

The orchestrator reads the refined single-char value, maps it to its integer (via the char-to-value table), finds the matching band, and emits the band label plus the raw numeric value.

#### 4.4.2 Adaptive Markdown Table Layout

The translated values are assembled into a markdown table. The table layout adapts based on variable count to optimise KV cache efficiency:

**≤50 variables: Column layout** (variables as columns)
```markdown
| user_frustration | task_completion | topic_relevance | confidence | ... |
|---|---|---|---|---|
| elevated (176/249) | 10% (26/249) | high (224/249) | low (48/249) | ... |
```
- Header row cached, only the value row recomputes from the first changed value.
- Best KV efficiency for small variable counts.

**50-200 variables: Chunked column tables** (groups of ~50 variables)
```markdown
**State chunk 1/4:**
| user_frustration | task_completion | ... (up to 50 columns) |
|---|---|---|
| elevated (176/249) | 10% (26/249) | ... |

**State chunk 2/4:**
| mood | engagement | ... (next 50 columns) |
|---|---|---|
| contentment (244/249) | moderate (128/249) | ... |
```
- Each chunk's header row caches independently.
- If only chunk 3 changed, chunks 1, 2, and 4+ are fully cached.
- Good balance between readability and KV efficiency.

**200+ variables: Row layout** (one variable per row)
```markdown
| Variable | Value |
|---|---|
| user_frustration | elevated (176/249) |
| task_completion | 10% (26/249) |
| topic_relevance | high (224/249) |
| ... | ... |
```
- Accept recompute cost — it's computed fresh once per turn after refinement converges anyway.
- Most readable for very large variable counts.

#### 4.4.3 KV Cache Properties of Each Layout

| Layout | When to use | Header cache | Value recompute | Total cost |
|---|---|---|---|---|
| Column | ≤50 vars | Header row cached across turns | Single value row recomputes from first change | ~5 tokens/var, minimal recompute |
| Chunked | 50-200 vars | Per-chunk headers cached independently | Only changed chunks recompute | ~5 tokens/var, partial recompute |
| Row | 200+ vars | Two-column header cached | All rows recompute | ~5 tokens/var, full recompute (once per turn) |

The generation context table is computed fresh **once per turn** after refinement converges. It is not recomputed during WIR refinement passes — only the micro-prompts are used during refinement. The table provides ~5 tokens per variable in readable form.

### 4.6 Header Fields

- `#WIRSTATE v1`: Format version identifier.
- `#COUNT N`: Number of state documents in the file.
- `#SHARED_PREFIX_TOKENS N`: Number of tokens in the shared system/refinement prompt prefix that is common to all states. The inference engine caches this prefix once and reuses it across all states.

### 4.7 Per-State Fields

- `====STATE entity_id=<id> hash=<sha256_prefix>====`: State boundary marker with entity ID and content hash (for cache invalidation).
- Standard WIR state document sections (`[IDENTITY]`, `[VARIABLES]`, `[GOALS]`, `[CONTEXT]`, `[RECENT]`, `[BELIEFS]`).
- `====END <id>====`: State terminator.

### 4.8 Shared Refinement Prompt

The batch file may optionally include a shared refinement prompt that applies to all state documents:

```
#WIRSTATE v1
#COUNT 200
#SHARED_PREFIX_TOKENS 82

====SHARED_PROMPT====
[SYSTEM] You are maintaining a living state document for a character
in a medieval village simulation. Update the document to reflect the
new events. Revise inconsistent beliefs. Compress stable information.
Preserve personality and core motivations. Output only the updated
state document.
====END SHARED_PROMPT====

====STATE entity_id=npc_001 ...====
...
```

The shared prompt is prepended to each state document during refinement. Its KV cache is computed once and reused for all states.

---

## 5. LLM-Assisted Compilation

### 5.1 When LLM Calls Are Needed

The compiler invokes an LLM for three tasks that cannot be performed by template expansion alone:

1. **Prose expansion:** Converting terse spreadsheet entries ("gruff but fair") into well-formed state document prose ("Gruff but fair master blacksmith with thirty years at the forge. Known for honest dealings and quality work, but prone to sharp words when interrupted.").

2. **Initial state generation:** Given a role description and parameters, generating a complete initial state document with coherent beliefs, goals, and context that are internally consistent.

3. **Validation:** Checking that compiled state documents are well-formed, that variable values are consistent with prose descriptions, and that required sections are populated.

### 5.2 LLM Endpoint Configuration

The compiler connects to an LM Studio (or any OpenAI-compatible) endpoint:

```yaml
# compiler-config.yaml
llm:
  endpoint: "http://localhost:1234/v1"
  model: "local-model"
  temperature: 0.7
  max_tokens: 2048
  # Rate limiting for large batches
  max_concurrent: 4
  delay_ms: 100
```

### 5.3 Compilation Pipeline

```
Input Files ──> Parse ──> Template Expand ──> [LLM Expand] ──> [LLM Validate] ──> Assemble ──> .wirstate
                  │              │                   │                │                │
                  │              │                   │                │                └─ Write batch file
                  │              │                   │                └─ Check well-formedness
                  │              │                   └─ Expand terse entries to prose
                  │              └─ Interpolate parameters into templates
                  └─ Read CSV/MD/YAML/JSON
```

Steps marked with `[brackets]` are optional and require an LLM endpoint. Without an LLM, the compiler produces state documents using only template expansion — functional but with terse prose sections.

### 5.4 LLM Expansion Prompt

For each state document that requires prose expansion, the compiler sends:

```
You are generating an initial state document for a WIR state refinement system.

Given the following structured data about an entity:
- ID: {entity_id}
- Role: {role}
- Personality: {personality}
- Goal: {current_goal}
- Context: {custom_context}
- Variables: {variables_dict}

Generate a complete state document with the following sections:
[IDENTITY] - 2-4 sentences establishing personality and role
[GOALS] - Bulleted list of current objectives
[CONTEXT] - Relevant background and situation
[BELIEFS] - What this entity currently believes about their situation
[RECENT] - Leave empty (no history yet)

The prose should be concise, specific, and internally consistent.
Do not include the [VARIABLES] section — that will be added separately.
Output only the sections, no commentary.
```

### 5.5 Batch Processing

For large compilations (hundreds or thousands of states), LLM expansion is the bottleneck. The compiler parallelises LLM calls up to `max_concurrent` and provides progress reporting:

```
Compiling 200 state documents...
  Parsed: 200/200 entities from village_npcs.csv
  Template expansion: 200/200 complete
  LLM expansion: 147/200 [=====>    ] 73% (est. 2m remaining)
  Validation: pending
```

---

## 6. Incremental Updates

### 6.1 New Input Injection

After initial compilation, the batch file needs to be updated with new input before each refinement pass. A separate tool (or mode of the same compiler) injects new input events into existing state documents:

```bash
wir-batch-inject --batch village.wirstate \
  --events events.json \
  --output village_updated.wirstate
```

Where `events.json` maps entity IDs to their new input:

```json
{
  "npc_001": "A merchant arrives offering high-quality iron at twice the normal price.",
  "npc_002": "Moonpetal flowers spotted in the eastern meadow by a traveller.",
  "npc_003": "Suspicious tracks found near the north gate at dawn.",
  "*": "The harvest festival is in three days."
}
```

The `"*"` key injects input to all entities — shared world events that every state document should process.

### 6.2 Post-Refinement Extraction

After the inference engine refines all states in the batch file, the updated batch file needs to be parsed back into individual state documents or structured data:

```bash
wir-batch-extract --batch village_refined.wirstate \
  --format csv \
  --output village_state.csv
```

This extracts the compact variables from all state documents into a CSV for monitoring, dashboard display, or application-layer consumption. The prose sections can also be extracted per-entity if needed.

---

## 7. Integration with llama-wir

### 7.1 Two-Phase Processing Mode

The `llama-wir` binary (or `llama-server`) accepts a `.wirstate` file and processes it in two phases:

```bash
llama-wir --model /path/to/model.gguf \
  --state-batch village.wirstate \
  --output village_refined.wirstate \
  --batch-parallel 8 \
  --refine-mode n-least-conf \
  --refine-n 4 \
  --max-passes 3 \
  --gen-table-layout auto
```

The engine:

**Phase A — Refinement (micro-prompt batch):**
1. Parses the `.wirstate` file into individual micro-prompts (one per variable per entity).
2. Caches shared Zone 1 instruction KV for variables with identical definitions across entities.
3. Batches all micro-prompts in a single weight-streaming pass.
4. Runs n-least-conf refinement — each micro-prompt independently decides whether its Zone 3 answer token should update.
5. Repeats until convergence (per-variable token change rate drops below threshold).

**Phase B — Generation context (translated table):**
1. Reads all converged Zone 3 answer tokens.
2. Auto-translates each single-char value to its readable form using the named band definitions from `.wirstate.meta`.
3. Assembles the translated values into a markdown table using the adaptive layout threshold (column/chunked/row based on variable count).
4. Injects the markdown table into the model's generation context.
5. The model generates its response with full awareness of all state dimensions in readable form.

### 7.2 Shared Prefix Optimisation

Micro-prompts that share the same Zone 1 instruction (e.g. all entities using the same `confidence_level` definition) benefit from a single prefix cache hit. For 200 entities with 10 shared variable definitions, this saves 199 redundant Zone 1 computations per variable definition — 1,990 saved computations total.

Zone 2 context can also be shared across all micro-prompts belonging to the same entity, since all variables for one entity see the same conversation context.

### 7.3 Variable-Length Pass Counts

Different micro-prompts may converge at different rates:

- A boolean variable with clear evidence: 1 pass (single-token convergence).
- A byte variable with ambiguous context: 2-3 passes.
- A variable whose correct value depends on external hook data contradicting its current value: 1-2 passes.

The engine tracks per-micro-prompt convergence and stops refining a micro-prompt when its answer token stabilises, even if other micro-prompts in the batch are still being refined. This prevents over-refinement of stable variables while allowing uncertain ones to converge.

### 7.4 Generation Context Injection

After refinement converges, the engine builds the generation context table:

1. For each variable, looks up the converged answer token's integer value.
2. Maps the integer to the named band label (e.g. 176 → "elevated" for frustration).
3. Assembles all variables into a markdown table using the adaptive layout:
   - `--gen-table-layout auto`: uses the threshold (≤50 column, 50-200 chunked, 200+ row).
   - `--gen-table-layout column|chunked|row`: forces a specific layout.
4. Prepends the table to the generation prompt.

The model sees readable values like `frustrated: elevated (176/249)` rather than `frustrated: Ñ`. This separation means refinement uses the most KV-efficient format (tiny micro-prompts with single-char answers) while generation uses the most informative format (readable labels with context).

---

## 8. Configuration

### 8.1 Compiler Configuration

```yaml
# wir-batch-compiler.yaml

input:
  format: auto          # auto-detect from file extension, or: csv, markdown, yaml, json
  file: input.csv
  template: null        # optional: markdown template for prose sections
  parameters: null      # optional: parameter sweep file

output:
  file: output.wirstate
  format: wirstate_v1

sections:
  # Map input columns/fields to state document sections
  identity_columns: [role, personality]
  goals_columns: [current_goal]
  context_columns: [custom_context, background]
  variables_columns: [confidence_level, task_priority, stressed]

llm:
  enabled: true
  endpoint: "http://localhost:1234/v1"
  model: "local-model"
  temperature: 0.7
  max_tokens: 2048
  max_concurrent: 4
  delay_ms: 100
  expand_prose: true    # expand terse entries to full prose
  validate: true        # validate compiled documents

encoding:
  # Rules for encoding column values as compact variables
  boolean_columns: [stressed, task_complete, context_stale]
  byte_columns: [confidence_level, task_priority, urgency_level]
  string_columns: [current_focus, blocked_on]
```

---

## 9. Example End-to-End Workflow

```
1. Author writes village_npcs.csv with 200 rows
   (one per NPC: id, role, personality, goals, initial variables)

2. Compiler reads CSV, expands prose via LLM, validates, writes village.wirstate
   $ wir-batch-compile --config compiler.yaml --input village_npcs.csv --output village.wirstate

3. Simulation tick: inject world events
   $ wir-batch-inject --batch village.wirstate --events tick_042.json --output village_tick042.wirstate

4. Refinement pass: llama-wir processes all 200 states
   $ llama-wir --model deepseek-v3.gguf --state-batch village_tick042.wirstate \
       --output village_tick042_refined.wirstate \
       --refine-mode n-least-conf --refine-n 4 --max-passes 3

5. Extract updated variables for game engine consumption
   $ wir-batch-extract --batch village_tick042_refined.wirstate --format csv --output npc_state.csv

6. Game engine reads npc_state.csv, updates NPC behaviours
   (stressed? Y triggers different dialogue; confidence_level < 100 triggers flee behaviour)

7. Next tick: repeat from step 3 with new events
```
