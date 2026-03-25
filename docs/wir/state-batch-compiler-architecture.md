# WIR State Batch Compiler — Architecture

**Status:** Architecture v0.1 — DRAFT
**Author:** Ed Holmes (Cranborne Audio)
**Date:** March 2026

---

## 1. System Overview

The system operates in two phases: **Phase A (Refinement)** compiles variables into micro-prompts and batches them for parallel refinement. **Phase B (Generation)** auto-translates refined values into a readable markdown table for the model's generation context.

```
                                    ┌─────────────────────────────────┐
                                    │     LM Studio / OpenAI API     │
                                    │   (prose expansion, validation) │
                                    └──────────────┬──────────────────┘
                                                   │ HTTP
                                                   │
┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──┴───────────┐   ┌─────────────┐
│   CSV    │──>│          │──>│  Template  │──>│     LLM      │──>│  Micro-     │
│   YAML   │   │  Parser  │   │  Expander  │   │   Expander   │   │  Prompt     │──> .wirstate
│ Markdown │──>│  Layer   │──>│            │──>│  (optional)  │──>│  Assembler  │   (micro-prompt
│   JSON   │   │          │   │            │   │              │   │             │    batch)
└──────────┘   └──────────┘   └────────────┘   └──────────────┘   └─────────────┘
                    │                                                     │
                    v                                                     v
              ┌──────────┐                                    ┌──────────────────┐
              │  Entity  │                                    │  Phase A:        │
              │  Registry│                                    │  .wirstate file  │
              │  (in-mem)│                                    │  (micro-prompts) │
              └──────────┘                                    └────────┬─────────┘
                                                                      │
                                                                      v
                                                              ┌───────────────┐
                                                              │  llama-wir    │
                                                              │  Batch Refine │
                                                              │  (all micro-  │
                                                              │   prompts in  │
                                                              │   one pass)   │
                                                              └───────┬───────┘
                                                                      │
                                                                      v
                                                              ┌───────────────┐
                                                              │  Value        │
                                                              │  Translator   │
                                                              │  (char->band  │
                                                              │   lookup)     │
                                                              └───────┬───────┘
                                                                      │
                                                                      v
                                                              ┌───────────────┐
                                                              │  Phase B:     │
                                                              │  Generation   │
                                                              │  Context      │
                                                              │  (markdown    │
                                                              │   table)      │
                                                              └───────────────┘
```

---

## 2. Component Architecture

### 2.1 Parser Layer

The parser layer normalises all input formats into a common intermediate representation: a list of `EntitySpec` records.

```
EntitySpec:
  id:          string              # unique identifier
  sections:    map<string, string> # section_name -> content
  variables:   map<string, Value>  # variable_name -> typed value
  metadata:    map<string, string> # compiler metadata (source row, etc.)
  needs_llm:   bool                # true if prose sections need expansion
```

Each input format has a dedicated parser:

```
┌─────────────────────────────────────────────────────────┐
│                     Parser Layer                         │
│                                                          │
│  ┌───────────┐  ┌───────────┐  ┌──────┐  ┌──────────┐  │
│  │ CSVParser  │  │ MDParser  │  │ YAML │  │ JSON     │  │
│  │           │  │           │  │Parser│  │ Parser   │  │
│  │ Row->Spec │  │ Block->   │  │Range │  │ Prompt-> │  │
│  │ with col  │  │ Spec with │  │->N   │  │ Spec     │  │
│  │ mapping   │  │ section   │  │Specs │  │ with     │  │
│  │           │  │ extraction│  │      │  │ shared   │  │
│  │           │  │           │  │      │  │ base     │  │
│  └───────────┘  └───────────┘  └──────┘  └──────────┘  │
│         │              │            │           │        │
│         └──────────────┴────────────┴───────────┘        │
│                         │                                │
│                    List<EntitySpec>                       │
└─────────────────────────────────────────────────────────┘
```

**CSV Parser:** Maps columns to sections/variables via configuration. Handles type detection (boolean columns contain only Y/N, byte columns contain integers 0-255, everything else is string).

**Markdown Parser:** Splits on `## entity_id` boundaries. Extracts `### Section` blocks as section content. Extracts `### Variables` blocks and parses key-value pairs with type inference.

**YAML Parameter Parser:** Generates the cartesian product (or random sample) of parameter ranges. Each combination produces one `EntitySpec` with template placeholders replaced by parameter values.

**JSON Prompt Parser:** Creates one `EntitySpec` per prompt entry, copying the shared base identity and variables and placing the prompt in `[GOALS]`.

### 2.2 Template Expander

Replaces placeholders in section content with values from the `EntitySpec`. This is pure string interpolation — no LLM needed.

```python
# Template: "You are a {role} with a {personality} demeanour."
# EntitySpec: {role: "blacksmith", personality: "gruff but fair"}
# Output:    "You are a blacksmith with a gruff but fair demeanour."
```

Templates can reference:
- Any section value: `{section.identity}`, `{section.goals}`
- Any variable: `{var.confidence_level}`, `{var.task_priority}`
- Any metadata: `{meta.source_row}`, `{meta.entity_id}`
- Computed expressions: `{'high' if var.confidence_level > 200 else 'low'}`

After template expansion, the `needs_llm` flag is set if any section content is shorter than a configurable threshold (default: 20 tokens) and `llm.expand_prose` is enabled. Terse entries like "gruff but fair" are flagged for LLM expansion; fully-written prose passes through.

### 2.3 LLM Expander

Expands terse section content into well-formed prose by calling an OpenAI-compatible endpoint. This is the only component that requires network access and an external model.

```
┌───────────────────────────────────────────────────┐
│                  LLM Expander                      │
│                                                    │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐  │
│  │ Prompt   │───>│ Request  │───>│ Response    │  │
│  │ Builder  │    │ Pool     │    │ Parser      │  │
│  │          │    │ (N conc) │    │             │  │
│  │ Entity + │    │ Rate     │    │ Extract     │  │
│  │ template │    │ limited  │    │ sections    │  │
│  │ -> prompt│    │ retries  │    │ from LLM    │  │
│  │          │    │          │    │ output      │  │
│  └──────────┘    └──────────┘    └─────────────┘  │
│                                                    │
│  Progress: [========>          ] 147/200  73%      │
└───────────────────────────────────────────────────┘
```

**Prompt Builder:** Constructs an expansion prompt from the `EntitySpec`, including the entity's structured data, the target section, and instructions for tone and length.

**Request Pool:** Manages concurrent HTTP requests to the LLM endpoint. Configurable concurrency limit (`max_concurrent`, default 4) and inter-request delay (`delay_ms`, default 100ms). Implements retry with exponential backoff for transient failures.

**Response Parser:** Extracts section content from the LLM response. Expects the response to contain section markers (`[IDENTITY]`, etc.) and parses accordingly. Falls back to treating the entire response as the target section if markers are absent.

### 2.4 Variable Encoder

Converts typed values from the `EntitySpec` into the compact variable format specified in the WIR State Refinement Method.

```
Encoding rules:
  bool   -> Y/N                                    (1 token)
  byte   -> single Unicode character (0-255)       (1 token, exactly)
  string -> "quoted value"                         (2-8 tokens, padded to max)
  int    -> clamped to 0-255, encode as byte
  float  -> mapped to 0-255 range, encode as byte
  enum   -> mapped to byte via lookup table
```

The safe single-character single-token values are drawn from:
- 87 ASCII printable (U+0021 to U+007E, minus 7 dangerous: `"<>\|` + backslash)
- 92 Latin-1 Supplement (U+00A0 to U+00FF, minus NBSP and soft hyphen)
- 68 Latin Extended-A (U+0100 to U+0143)
- **Total: 249 safe values** (conservative mode: 240, also excluding `# ' / = [ ] { }` + backtick)

Each character is guaranteed to tokenise as exactly 1 token across Qwen and Mistral tokeniser families. 7 characters are excluded because they risk triggering special model behaviour (EOS sequences, escape parsing, invisible characters). 249 levels provides ~0.4% granularity.

The encoder also generates a companion lookup table file (`.wirstate.meta`) that records the mapping for each variable (enabling the application layer to decode byte values back to their original scale) and the char-to-value lookup table:

```json
{
  "confidence_level": {"type": "byte", "scale": "linear", "min": 0, "max": 100, "unit": "%"},
  "task_priority": {"type": "byte", "scale": "enum", "values": ["low", "medium", "high", "critical"]},
  "stressed": {"type": "bool"},
  "_char_table": ["!", "\"", "#", "$", "...(256 chars)...", "ń"]
}
```

**Model-specific token tables (late goal):** The encoder currently uses a hardcoded 256-char table confirmed for Qwen/Mistral. It should eventually support model-specific token tables by querying the GGUF vocab to find all single-character tokens and building the optimal mapping. This would be implemented by loading the model's vocabulary, iterating all tokens, selecting those that decode to a single Unicode character, and constructing a 256-entry mapping sorted by code point or frequency.

### 2.5 Micro-Prompt Assembler

Combines all processed `EntitySpec` records into a batch of micro-prompts in the final `.wirstate` file. Each variable for each entity becomes its own independent micro-prompt.

```
┌──────────────────────────────────────────────────────┐
│              Micro-Prompt Assembler                    │
│                                                        │
│  1. Write header (#WIRSTATE, #COUNT, #FORMAT)          │
│  2. For each EntitySpec:                               │
│     For each variable in the entity:                   │
│       a. Write #MICROPROMPT header (var + entity ID)   │
│       b. Write Zone 1: variable instruction (~20 tok)  │
│       c. Write Zone 2: conversation context (~50 tok)  │
│       d. Write Zone 2b: external state (~10 tok)       │
│       e. Write Zone 3: encoded answer value (1 tok)    │
│  3. Identify shared Zone 1 instructions across         │
│     entities (same variable def = same Zone 1 KV)      │
│  4. Compute per-micro-prompt token estimate (~80 tok)   │
│  5. Write metadata file (.wirstate.meta) including     │
│     named value bands for auto-translation             │
└──────────────────────────────────────────────────────┘
```

The assembler generates one micro-prompt per variable per entity. For 200 entities with 10 variables each, this produces 2,000 micro-prompts, each ~80 tokens, all batched in a single weight-streaming pass. The assembler marks which Zone 1 instructions are shared across entities so the inference engine can cache them once and reuse.

### 2.6 Value Translator

A new component that takes refined single-char values from Phase A and produces readable key-value pairs for Phase B. The translator requires no LLM calls — it is a pure lookup operation.

```
┌──────────────────────────────────────────────────────┐
│                  Value Translator                      │
│                                                        │
│  Input:                                                │
│    Refined Zone 3 values (single-char tokens)          │
│    Variable definitions with named bands               │
│                                                        │
│  Process:                                              │
│    1. Map single-char token -> integer (char table)    │
│    2. Map integer -> named band label (band defs)      │
│    3. Format: "label (value/max)"                      │
│                                                        │
│  Example:                                              │
│    Ñ -> 176 -> "elevated" (frustration bands)          │
│    Output: "elevated (176/249)"                        │
│                                                        │
│  Band definition (from .wirstate.meta):                │
│    frustration:                                        │
│      bands:                                            │
│        0-50: "calm"                                    │
│        51-120: "mild"                                  │
│        121-180: "elevated"                             │
│        181-249: "furious"                              │
│                                                        │
│  Booleans: Y -> "yes", N -> "no" (or custom labels)   │
│  Strings: passed through unchanged                     │
└──────────────────────────────────────────────────────┘
```

### 2.7 Generation Context Assembler

Builds the markdown table from translated values for injection into the model's generation context.

```
┌──────────────────────────────────────────────────────┐
│           Generation Context Assembler                 │
│                                                        │
│  Input:                                                │
│    Translated key-value pairs from Value Translator    │
│    Variable count (determines table layout)            │
│                                                        │
│  Layout selection (adaptive threshold):                │
│    ≤50 vars:   Column layout (vars as columns)         │
│    50-200 vars: Chunked column tables (~50 per chunk)  │
│    200+ vars:  Row layout (one var per row)            │
│                                                        │
│  Output: Markdown table string                         │
│    Injected into model's generation prompt             │
│    Computed once per turn after refinement converges    │
│    ~5 tokens per variable in readable form             │
│                                                        │
│  KV cache properties:                                  │
│    Column:  Header row cached, value row recomputes    │
│    Chunked: Per-chunk headers cached independently     │
│    Row:     Full recompute (acceptable, once per turn) │
└──────────────────────────────────────────────────────┘
```

---

## 3. Event Injection Architecture

### 3.1 Event Injector

A separate tool (or mode of the compiler) that modifies an existing `.wirstate` file by appending new input to specified states.

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ events   │───>│   Event      │───>│  Updated     │
│ .json    │    │   Injector   │    │  .wirstate   │
│          │    │              │    │              │
└──────────┘    │  Parse batch │    └──────────────┘
                │  Match events│
┌──────────┐    │  Append input│
│ existing │───>│  Recompute   │
│ .wirstate│    │  hashes      │
└──────────┘    └──────────────┘
```

Event matching:
- **By entity ID:** `"npc_001": "event text"` — inject to specific entity.
- **By wildcard:** `"*": "event text"` — inject to all entities.
- **By variable filter:** `"?stressed=Y": "event text"` — inject to entities matching a variable condition.
- **By section content:** `"?identity~blacksmith": "event text"` — inject to entities whose identity section contains "blacksmith".

### 3.2 Extraction Pipeline

Reads a refined `.wirstate` file and extracts structured data.

```
┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Refined     │───>│  Extractor   │───>│  CSV     │
│  .wirstate   │    │              │    │  JSON    │
│              │    │  Parse batch │    │  SQLite  │
└──────────────┘    │  Extract vars│    └──────────┘
                    │  Extract text│
                    │  Diff vs prev│
                    └──────────────┘
```

Output formats:
- **CSV:** One row per entity, columns for each variable and optionally section summaries.
- **JSON:** Full structured representation of all entities.
- **SQLite:** Write to a database table for query-based analysis.
- **Diff:** Compare two `.wirstate` files (before/after refinement) and output per-entity change reports.

---

## 4. Data Flow: Complete Cycle

```
                  ┌──────────────────────────────────────────┐
                  │           Compile (once)                   │
                  │                                            │
 CSV/MD/YAML ────>│  Parse -> Expand -> LLM -> Micro-Prompt   │───> initial.wirstate
                  │                             Assembler      │    (micro-prompt batch)
                  └──────────────────────────────────────────┘
                                    │
                                    v
                  ┌──────────────────────────────────────────┐
            ┌────>│         Inject Events                     │
            │     │                                            │
            │     │  batch.wirstate + events.json               │───> updated.wirstate
            │     │  (update Zone 2/2b in each micro-prompt)    │
            │     └──────────────────────────────────────────┘
            │                       │
            │                       v
            │     ┌──────────────────────────────────────────┐
            │     │   Phase A: llama-wir Batch Refine         │
            │     │                                            │
            │     │   All micro-prompts in single              │
            │     │   weight-streaming pass                    │───> refined.wirstate
            │     │   --refine-mode n-least-conf               │    (converged values)
            │     │   Per-variable convergence                 │
            │     └──────────────────────────────────────────┘
            │                       │
            │                       v
            │     ┌──────────────────────────────────────────┐
            │     │   Phase B: Translate + Generation Table    │
            │     │                                            │
            │     │   Value Translator: char -> band label     │
            │     │   Generation Context Assembler:            │
            │     │     adaptive markdown table                │───> Generation
            │     │     (column/chunked/row)                   │    context for
            │     │   Injected into model's generation prompt  │    model response
            │     └──────────────────────────────────────────┘
            │                       │
            │                       v
            │     ┌──────────────────────────────────────────┐
            │     │         Extract State                      │
            │     │                                            │
            │     │  refined.wirstate -> state.csv              │───> Application
            │     │                    -> state.json            │    (game engine,
            │     │                    -> state.db              │     dashboard,
            │     └──────────────────────────────────────────┘     controller)
            │                       │
            │                       │ refined.wirstate becomes
            │                       │ the new batch for next cycle
            └───────────────────────┘
```

---

## 5. Concurrency Model

### 5.1 Compilation Phase

During compilation, the only concurrent operations are LLM expansion calls. These are IO-bound (waiting for HTTP responses) and benefit from asyncio-style concurrency:

```
Main thread:
  Parse all inputs (fast, sequential)
  Template expand all (fast, sequential)
  Submit LLM expansion tasks to pool
  Await all completions
  Assemble output (fast, sequential)

LLM pool (N workers):
  Worker 1: expand entity_001, entity_005, entity_009, ...
  Worker 2: expand entity_002, entity_006, entity_010, ...
  Worker 3: expand entity_003, entity_007, entity_011, ...
  Worker 4: expand entity_004, entity_008, entity_012, ...
```

### 5.2 Inference Phase

The inference engine (llama-wir) handles parallelism internally via batch scheduling. The compiler's output is a single file; the engine decides how to parallelise across available compute.

On weight-streaming hardware, all states in the batch are processed in a single weight-streaming pass: model weights are loaded once, and the batch of states is processed as a batch of prefill sequences sharing the same weight stream. This is the optimal configuration — the cost of refining N states is approximately the cost of refining 1 state (amortised weight loading) plus O(N) for the per-state KV computation.

---

## 6. Error Handling

### 6.1 Parse Errors

- Missing required column (`entity_id`): Fatal. Compiler exits with error listing the missing column.
- Malformed CSV/JSON/YAML: Fatal. Parser reports line number and error.
- Missing section in markdown: Warning. Empty section created.

### 6.2 LLM Errors

- Endpoint unreachable: Retry 3 times with exponential backoff, then fall back to template-only expansion for that entity. Log warning.
- Malformed LLM response (no section markers, truncated): Retry once. On second failure, use raw LLM output as `[CONTEXT]` section. Log warning.
- Rate limit (429): Exponential backoff with jitter. Reduce concurrency to 1 if persistent.

### 6.3 Encoding Errors

- Integer out of byte range (>255): Clamp to 255. Log warning.
- Unrecognised variable type: Default to short string encoding. Log warning.

### 6.4 Assembly Errors

- Duplicate entity IDs: Fatal. List duplicates.
- Empty state document (no sections populated): Warning. Include in output with empty sections.

---

## 7. File Format Versioning

The `.wirstate` format includes a version header (`#WIRSTATE v1`). Future versions may add:

- **v2:** Compressed sections (zstd per-state for large batches).
- **v2:** Binary variable encoding (raw bytes instead of single Unicode character text).
- **v2:** Embedded KV cache snapshots (cache data inline with state for single-file deployment).

The compiler and inference engine negotiate format version. A v2 engine can read v1 files. A v1 engine rejects v2 files with a clear error message.
