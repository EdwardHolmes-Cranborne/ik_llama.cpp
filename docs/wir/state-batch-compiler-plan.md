# WIR State Batch Compiler — Implementation Plan

**Status:** Plan v0.2 — DRAFT
**Author:** Ed Holmes (Cranborne Audio)
**Date:** March 2026

---

## 1. Core Design Constraint

The wirstate document serves two roles simultaneously:

1. **Refinement target:** WIR passes update the mutable answer fields.
2. **LLM context:** After refinement, the whole document is read by the model as context before it generates its actual response.

This dual role constrains the format. It must be:
- Machine-parseable (the inference engine needs to identify mutable vs immutable regions)
- LLM-readable (the model reads it as coherent context for generation)
- KV-cache-efficient (only mutable tokens trigger recomputation)

---

## 2. Variable Anatomy

Every variable is refined as its own independent **micro-prompt** — a tiny conversation with four zones. The zone layout is **per micro-prompt**, not one big document with all variables stacked together.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    Single Micro-Prompt (~80 tokens)                             │
│                                                                                │
│ Zone 1: INSTRUCTION    │ Zone 2: CONTEXT     │ Zone 2b: EXTERNAL │ Zone 3:     │
│ (~20 tokens, fixed)    │ (~50 tokens,        │ (~10 tokens,      │ ANSWER      │
│ Never changes. KV      │ per-turn)           │ per-turn)         │ (1 token)   │
│ cached forever.        │ Changes when convo  │ Hook-injected     │ Mutable.    │
│                        │ context changes.    │ ground truth.     │ WIR target. │
├────────────────────────┼─────────────────────┼───────────────────┼─────────────┤
│ "User frustration      │ "User just said:    │ errors_last_5m:   │ Ñ           │
│  level (0=calm,        │  'third time asking │ 042               │ (=176)      │
│  249=furious):"        │   the same thing'"  │                   │             │
└────────────────────────┴─────────────────────┴───────────────────┴─────────────┘
   KV cached forever        KV recomputed on       KV recomputed       1 token
                             new conversation       when hooks fire     WIR target
                             input only
```

**Why micro-prompts, not one big document:** In a single large document with 200 variables, the last variable's answer is thousands of tokens away from the first variable's instruction. Attention quality degrades over distance. With micro-prompts, each variable's instruction is always ~80 tokens from its answer — perfect attention proximity. All 200 micro-prompts are batched in a single weight-streaming pass, so the total compute cost is the same.

### 2.1 Zone Properties (Per Micro-Prompt)

| Zone | Contents | Changes when | KV cache behaviour | Approx. tokens |
|---|---|---|---|---|
| Zone 1: INSTRUCTION | Variable name, type, range, interpretation | Never (compile time) | Computed once, cached permanently | ~20 |
| Zone 2: CONTEXT | Conversation excerpt relevant to this variable | New conversation input arrives | Recomputed per-turn, cached across WIR passes | ~50 |
| Zone 2b: EXTERNAL | Hook-injected ground truth relevant to this variable | When hooks fire (per-turn) | Same as Zone 2 | ~10 |
| Zone 3: ANSWER | Single mutable value (byte/bool/string) | Every WIR refinement pass | Recomputed each pass (1 token for byte/bool) | 1 |

### 2.2 KV Cache Efficiency Per Micro-Prompt

Within each ~80-token micro-prompt:
- Zone 1 KV: **never recomputed** (prefix cache hit, loaded from disk)
- Zone 2 KV: **recomputed once per conversation turn**, then cached for all WIR passes
- Zone 2b KV: **recomputed when hooks fire** (typically per-turn), then cached for all WIR passes
- Zone 3 KV: **recomputed each WIR pass** — but it's only 1 token per variable

For 100 variables: 100 micro-prompts at ~80 tokens each = ~8,000 tokens total, all batched in a single pass. Per WIR pass, only 100 tokens recompute (one Zone 3 answer per micro-prompt). That's near-instant.

### 2.3 As Context for Generation (Phase B)

After refinement, the compact single-char tokens are **not** used directly as generation context. Instead, the orchestrator auto-translates them into readable values and assembles a markdown table:

During refinement (Phase A), the model sees:
```
user_frustration: Ñ    (compact, 1 token)
```

For generation (Phase B), the model sees:
```
| user_frustration | task_completion | topic_relevance | confidence | ... |
|---|---|---|---|---|
| elevated (176/249) | 10% (26/249) | high (224/249) | low (48/249) | ... |
```

The translation is a simple lookup using named value bands defined at compile time — no LLM call needed. The model generates its response with full awareness of all state dimensions in human-readable form. See the spec document section 4.4 for details on the adaptive table layout (column/chunked/row based on variable count).

---

## 3. Revised Phase Plan

### Phase 0: Format Validation (1-2 days)

**Goal:** Prove the two-phase architecture works before building any tooling.

**Method:** Hand-write micro-prompts for 5-10 variables. Load them as prompts in LM Studio or llama-cli. Verify:
1. The model correctly reads the micro-prompt format (instruction + context + external + answer) and can update the single answer token.
2. The micro-prompt format works — each variable refined independently in its own tiny prompt.
3. Auto-translation produces readable output — single-char tokens map correctly to named band labels via the lookup table.
4. The markdown table (generated from translated values) is understood by the model as generation context — the model demonstrates awareness of all variable values when generating a response.
5. The 249-char single-token encoding is confirmed with the specific target model's tokeniser.

**Encoding answer (confirmed):** The 249-value single-token safe table uses 87 ASCII printable + 92 Latin-1 Supplement + 68 Latin Extended-A + 2 bonus = 249 characters, each guaranteed to be a single token across Qwen and Mistral tokeniser families. Phase 0 should still validate this against the specific target model by tokenising all 249 characters and confirming single-token behaviour.

**Deliverable:** A validated two-phase format specification: (a) micro-prompt format confirmed for refinement, (b) auto-translation confirmed for readable output, (c) markdown table confirmed as effective generation context, (d) tokeniser behaviour confirmed for the 249-char table. This gates everything else.

**Exit criteria:** Hand-crafted micro-prompts where the model correctly reads and updates 5+ variables (one per micro-prompt, using single-character byte encoding), and a hand-assembled markdown table (from auto-translated values) that the model uses effectively as generation context.

---

### Phase 1: Minimal Compiler + Single-Variable WIR Test (3-5 days)

**Goal:** CSV in, wirstate out (micro-prompt batch), prove single-variable refinement works end-to-end in llama-wir, prove auto-translation and generation context table work.

**Tasks:**
1. Python script: read CSV, emit wirstate as a batch of micro-prompts using single-token encoding.
   - CSV columns: `var_name, instruction, type (bool/byte/string), initial_value, bands (optional)`
   - Output: wirstate with one micro-prompt per variable, each containing Zone 1 (~20 tokens), Zone 2 (placeholder context), Zone 2b (empty), Zone 3 (initial value as single Unicode character)
2. Python script: read refined wirstate, extract Zone 3 values, auto-translate to readable form using named band definitions, emit CSV with both raw and translated values.
3. Python script: assemble translated values into a markdown table using the adaptive layout threshold (column for ≤50 vars, chunked for 50-200, row for 200+).
4. Minimal llama-wir modification: load wirstate, parse micro-prompts, identify Zone 3 token position in each micro-prompt, run n-least-conf refinement targeting only Zone 3 tokens.

**Validation experiment:**
- Compile 5 variables with deliberately wrong initial values and named band definitions.
- Inject a conversation context that makes the correct values obvious.
- Run 1-3 WIR passes with n-least-conf.
- Extract values, auto-translate, and verify they converged to sensible answers.
- Generate a markdown table from the translated values and verify the model uses it effectively as generation context.

**What this proves:** The model understands the micro-prompt format, WIR can target only the answer tokens, single-token variables converge in 1 pass, auto-translation produces readable output, and the generation context table works.

**Deliverables:**
- `tools/wir-batch-compile/compile.py` (minimal, CSV only, micro-prompt output)
- `tools/wir-batch-compile/extract.py` (minimal, CSV out with auto-translation)
- `tools/wir-batch-compile/gen_context.py` (generation context table assembler)
- Modifications to `examples/wir/wir.cpp` for micro-prompt-aware refinement
- Test results documenting convergence and generation quality

---

### Phase 2: KV Cache Validation (3-5 days)

**Goal:** Prove the micro-prompt KV cache strategy actually works — that Zone 1 and Zone 2 caches are reusable across passes and turns within each micro-prompt.

**Tasks:**
1. Instrument llama-wir to log KV cache hit/miss per zone per micro-prompt during refinement.
2. Run a multi-pass refinement (3 passes) over 10 micro-prompts. Verify Zone 1 KV is computed once per micro-prompt and reused for passes 2 and 3.
3. Simulate a new conversation turn: change Zone 2 context in each micro-prompt, keep Zone 1 and Zone 3 from previous refinement. Verify Zone 1 cache hits, Zone 2 recomputes, Zone 3 recomputes.
4. Measure actual token counts per zone per micro-prompt and per-pass compute cost.

**Validation experiment:**
- 10 micro-prompts (~80 tokens each = ~800 tokens total), 3 WIR passes, 2 conversation turns.
- Log: tokens computed per pass per micro-prompt, wall-clock time per pass.
- Expected: Pass 1 computes all zones in all micro-prompts. Pass 2-3 compute only Zone 3 (1 token each) per micro-prompt. Turn 2 pass 1 computes Zone 2 + Zone 2b + Zone 3 per micro-prompt. Turn 2 passes 2-3 compute only Zone 3.

**What this proves:** The micro-prompt format delivers the KV efficiency we're claiming. Because each micro-prompt is independent and Zone 3 is a single token at the end, there are no cross-variable cache invalidation issues.

**Critical risk (largely mitigated by micro-prompts):** In the old single-document layout, a changed Zone 3 token could invalidate KV for subsequent variables. With micro-prompts, each variable is independent — changing one micro-prompt's Zone 3 has zero effect on other micro-prompts. The only remaining risk is string-type variables where Zone 3 might be 2-8 tokens; fixed-width padding handles this.

**Mitigation:** Fixed-width answer slots for string variables. Pad string answers to their maximum token count. Byte answers are always exactly 1 token (single Unicode character). Bool answers are always 1 token (Y/N).

**Deliverables:**
- KV cache instrumentation in llama-wir (per micro-prompt tracking)
- Benchmark results documenting cache hit rates per zone per micro-prompt
- Comparison: micro-prompt KV efficiency vs theoretical single-document approach

---

### Phase 3: Batch Scaling (3-5 days)

**Goal:** Prove that N micro-prompts scale efficiently, test the adaptive table layout thresholds, and verify generation context quality at scale.

**Tasks:**
1. Scale from 10 to 50 to 100 to 200 variables (= 10 to 200 micro-prompts) in a single wirstate.
2. Measure: total token count (micro-prompts at ~80 tokens each), refinement time per pass, KV cache sizes.
3. Test the adaptive table layout threshold: at 50 variables verify column layout works, at 100 verify chunked layout works, at 200+ verify row layout works. Measure KV cache efficiency of each layout.
4. Test generation quality: after refinement, auto-translate values and build the generation context table. Have the model generate a response. Verify the response demonstrates awareness of the variable values from the readable table.
5. Test variable interaction: set up variables whose correct values depend on each other (e.g. `task_complete` should flip to Y when `task_completion_pct` reaches 249). Verify WIR converges these related variables within 2-3 passes even though they are in separate micro-prompts.

**Validation experiment:**
- 100 variables (= 100 micro-prompts, ~8,000 tokens total), mix of bool/byte/string types.
- Each variable has named band definitions for auto-translation.
- Inject context describing a specific scenario.
- Refine with n-least-conf, N=10 (fix 10 worst per pass), 5 passes max.
- Auto-translate all values, build chunked column table (100 vars = 2 chunks of 50).
- Generate a response using the markdown table as context, verify quality.

**What this proves:** The micro-prompt architecture scales linearly, the adaptive table layout produces usable generation context, and variables interact correctly across separate micro-prompts.

**Deliverables:**
- Scaling benchmarks (10/50/100/200 micro-prompts)
- Adaptive table layout comparison (column vs chunked vs row)
- Generation quality evaluation at each scale point
- Variable interaction convergence analysis (cross-micro-prompt dependencies)

---

### Phase 4: Multi-Entity Batching (5-7 days)

**Goal:** Multiple entities (NPCs, agents, sessions) each with their own wirstate, processed in a single weight-streaming pass.

**Tasks:**
1. Define batch format: multiple wirstates concatenated with entity boundaries.
2. Implement shared Zone 1 prefix: if all entities share the same variable definitions, the instruction block is identical and its KV cache is computed once, copied N times.
3. Each entity has its own Zone 2 (different context) and Zone 3 (different values).
4. Implement in llama-wir: batch processing with shared prefix cache.
5. Compiler support: CSV with entity_id column, one wirstate per entity in the batch.
6. Extraction: per-entity CSV output.

**Validation experiment:**
- 20 entities, 10 variables each, shared variable definitions.
- Each entity has different context and initial values.
- Single WIR refinement pass across all 20 entities.
- Verify: shared prefix computed once, per-entity values refined independently.
- Measure: amortised cost per entity vs single-entity cost.

**What this proves:** Batch processing amortises weight-streaming cost across entities.

**Deliverables:**
- Batch wirstate format
- llama-wir batch processing mode
- Compiler with entity support
- Batch scaling benchmarks (1/5/10/20/50 entities)

---

### Phase 5: Full Toolchain (5-7 days)

**Goal:** Production-quality compiler with multiple input formats, event injection, and extraction.

**Tasks:**
1. Additional parsers: markdown, YAML parameter sweeps, JSON prompt lists.
2. Event injection: modify Zone 2 context and optionally Zone 3 values for specific entities.
3. Full extraction: CSV, JSON output with variable decoding (byte -> human-readable via lookup table).
4. Diff tool: compare before/after wirstate files.
5. Configuration file for variable definitions, encoding tables, section templates.
6. Optional LLM expansion for terse entries (via LM Studio endpoint).

**Deliverables:**
- Complete `tools/wir-batch-compile/` toolchain
- Sample files for each input format
- End-to-end demo: compile NPCs -> inject events -> refine -> extract -> display

---

## 4. Encoding Decision: The ASCII Table

The choice of answer encoding is critical. It must be:
- **Single-token** in the target model's tokeniser (or fixed-token-count)
- **Ordered** so the model understands relative magnitude
- **Readable** so the concatenated document makes sense as LLM context

### 4.1 Chosen Encoding: Single-Token Unicode Characters (Safe Table)

Each byte variable is encoded as a single Unicode character guaranteed to tokenise as exactly 1 token. The character pool is drawn from three ranges confirmed single-token across Qwen and Mistral families:

| Range | Code points | Total | Safe | Excluded |
|---|---|---|---|---|
| ASCII printable | U+0021 – U+007E | 94 | 87 | 7 (`"<>\|` + backslash) |
| Latin-1 Supplement | U+00A0 – U+00FF | 94 | 92 | 2 (NBSP, soft hyphen) |
| Latin Extended-A | U+0100 – U+0143 | 68 | 68 | 0 |
| **Total** | | **256** | **249** | **7** |

7 characters are excluded because they have special meaning to the model or the wirstate format (EOS-related `<>|`, escape `\`, invisible NBSP/SHY, string delimiter `"`). 9 borderline characters (`# ' / = [ ] { }` + backtick) are included by default but can be excluded in conservative mode (240 values).

**249 safe values** provides ~0.4% granularity — more than sufficient for all practical variable ranges.

### 4.2 Encoding Rules

- **Booleans:** `Y` / `N` — 1 token, self-explanatory.
- **Byte values:** Single Unicode character from the 256-char table. The Zone 1 instructions include the char-to-value lookup table so the model knows which character means which value. The `.wirstate.meta` file includes the same table for application code.
- **Short strings:** Quoted, max 8 tokens: `"debugging auth"`. Fixed-width pad with spaces if needed.

### 4.3 Phase 0 Validates This

The 249-char safe table is confirmed for Qwen/Mistral families. Phase 0 still validates the table against the specific target model's tokeniser by tokenising all 249 characters and confirming each is a single token. Phase 0 should also verify that none of the safe characters appear in the model's special token list (EOS, BOS, chat template tokens). If any character fails validation, the compiler falls back to the conservative 240-char table or identifies replacements from the model's single-character token set.

**Per-variable n-least-conf:** Because each variable is its own micro-prompt with exactly 1 mutable token (Zone 3), n-least-conf operates trivially: each micro-prompt independently decides whether its single answer token should update. There is no need to partition a global token list by variable slot — each micro-prompt's Zone 3 is the only mutable token in its context. This simplifies the `wir.cpp` implementation: each micro-prompt in the batch is treated as an independent refinement unit with one mutable position.

**Model-specific token tables (late goal — Phase 5+):** The compiler should eventually auto-generate lookup tables matched to the loaded model's tokeniser. This involves querying the GGUF vocab to find all single-character tokens and building the optimal 256-char mapping. For now, the hardcoded table covering Qwen/Mistral families is sufficient.

---

## 5. Implementation Language and Location

Python 3.10+ for the compiler toolchain. Lives at `tools/wir-batch-compile/`.

C++ modifications to `examples/wir/wir.cpp` for zone-aware refinement and batch processing.

```
tools/wir-batch-compile/
├── compile.py              # CSV/MD/YAML/JSON -> .wirstate (micro-prompt batch)
├── inject.py               # Update Zone 2/2b context in micro-prompts
├── extract.py              # .wirstate -> CSV/JSON (with auto-translation)
├── gen_context.py          # Build generation context markdown table
├── diff.py                 # Compare before/after
├── core/
│   ├── entity.py           # EntitySpec, VariableSpec dataclasses
│   ├── encoder.py          # Value encoding (bool/byte/string)
│   ├── microprompt.py      # Micro-prompt builder (zones per variable)
│   ├── translator.py       # Value Translator (char -> band label)
│   ├── table_builder.py    # Adaptive markdown table layout
│   ├── assembler.py        # .wirstate writer (micro-prompt batch)
│   └── reader.py           # .wirstate parser
├── parsers/
│   ├── csv_parser.py
│   ├── markdown_parser.py
│   ├── yaml_parser.py
│   └── json_parser.py
├── llm/
│   ├── client.py           # OpenAI-compatible async client
│   └── prompts.py          # Expansion prompts
├── samples/
│   └── ...
└── tests/
    ├── test_encoder.py
    ├── test_microprompt.py
    ├── test_translator.py
    ├── test_table_builder.py
    └── test_roundtrip.py
```

---

## 6. Dependencies

```
# requirements.txt (Phase 1-3: only csv and core)
pyyaml>=6.0             # Phase 5: YAML parser support

# Phase 5 additions
openpyxl>=3.1           # XLSX support
httpx>=0.27             # LLM endpoint calls
jinja2>=3.1             # Template expansion
```

Minimal deps for early phases. Just Python stdlib for Phase 1.

---

## 7. Risk Register

| Risk | Impact | Mitigation | Phase |
|---|---|---|---|
| Target model's tokeniser doesn't match the 249-char safe table | Some characters tokenise as multi-token or are special tokens | Phase 0 validates all 249 chars; fall back to conservative 240-char table; compiler will eventually auto-generate model-specific tables | 0 |
| KV cache invalidated by Zone 3 changes during WIR | No cache efficiency, defeats the purpose | Micro-prompts largely mitigate this (each is independent); fixed-width answer slots for string variables; verify with instrumentation | 2 |
| Model doesn't understand the micro-prompt format | Can't read its own state, can't refine | Phase 0 validates with hand-crafted micro-prompts | 0 |
| Model doesn't understand the generation context markdown table | Can't use refined state for response generation | Phase 0 validates with hand-assembled table; test all three layouts (column/chunked/row) | 0 |
| Variable interactions cause oscillation across micro-prompts | Related variables in separate micro-prompts flip-flop across passes | n-least-conf naturally handles this (fixes worst first); shared context in Zone 2 helps; test in Phase 3 | 3 |
| Batch scaling hits context window limits | Can't fit 200 entities' micro-prompts in one pass | Each micro-prompt is only ~80 tokens; 200 vars = ~16K tokens; shared Zone 1 dedup reduces further | 4 |
| Auto-translation band definitions are ambiguous | Named bands don't match model's understanding | Validate band labels in Phase 0; allow custom band definitions per variable | 1 |
| Adaptive table layout threshold wrong | Column layout too wide for 50 vars, row layout wasteful for 150 | Test all three layouts at boundary points (49, 50, 51, 199, 200, 201) in Phase 3 | 3 |
| String answer width varies across passes | Cache invalidation within micro-prompt | Fixed-width padding with space fill | 2 |

---

## 8. Success Criteria

| Phase | Criterion |
|---|---|
| 0 | Model correctly reads and updates 5+ variables in hand-crafted micro-prompts; auto-translated markdown table works as generation context |
| 1 | 5 deliberately-wrong variables converge to correct values in ≤3 WIR passes; auto-translation produces correct band labels; generation context table is coherent |
| 2 | Zone 1 KV cache hit rate >99% across passes per micro-prompt; Zone 2 hit rate >99% within a turn |
| 3 | 100 micro-prompts (~8K tokens) refine in <500ms per pass on RTX 5090; adaptive table layout produces quality generation context at 50/100/200 variable scales |
| 4 | 20 entities processed in single pass; amortised cost per entity <50% of single-entity cost |
| 5 | Full toolchain: compile 50 entities from CSV, inject events, refine, auto-translate, generate context table, extract, verify |
