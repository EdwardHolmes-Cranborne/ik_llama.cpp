# WIR Tiered Model Architecture

## T1/T2/T3 Cooperative Inference with Online Adaptation

**Status:** Specification v0.1 — DRAFT
**Author:** Ed Holmes (Cranborne Audio)
**Date:** March 2026
**Parent:** WIR State Refinement Method v0.1

---

## 1. Overview

The WIR Cognitive Runtime uses three tiers of model working cooperatively:

| Tier | Role | Size | Latency | Runs when |
|---|---|---|---|---|
| **T1** | Classifier / sensor | 0.5-3B | <10ms | Every tick (continuous) |
| **T2** | Responder / actor | 7-14B | 50-200ms | On events, user input |
| **T3** | Refiner / judge | 27-70B+ | 0.5-4s | WIR refinement passes |

T1 runs continuously, classifying inputs, updating compact state variables, and detecting events that require T2 or T3 attention. T2 handles interactive responses and mid-complexity reasoning. T3 runs WIR refinement passes over state documents, correcting T1/T2 outputs and maintaining long-term coherence.

The tiers are not a fallback chain — they operate concurrently on different aspects of the same state. T1 provides real-time classification. T2 provides interactive response. T3 provides deep refinement. Each tier reads and writes to the shared wirstate.

---

## 2. T1: Continuous Classifier

T1 is a small model (0.5-3B) that runs on every tick of the system clock. Its job is fast classification:

- Read incoming data (user input tokens, sensor values, game events)
- Update compact state variables (byte values, booleans) in the wirstate
- Flag events that need T2 attention (e.g., user asked a question)
- Flag state inconsistencies that need T3 refinement

T1 processes each variable as a micro-prompt (~80 tokens). At 0.5-3B model size, a single micro-prompt completes in <10ms. A full sweep of 100 variables completes in <100ms with batching.

### 2.1 T1 Model Strategy (Three Layers)

T1's capabilities come from three composable layers:

1. **Base model** — General language understanding. A pretrained 0.5-3B model. Never modified during deployment.
2. **Specialist fine-tune** — Task-specific classification skill. Trained offline on accumulated logs from previous deployments. Updated periodically (weekly/monthly) as more training data accumulates. Provides the core classification accuracy for the target domain.
3. **Online LoRA** — Deployment-specific adaptation. Trained continuously during operation via T3 corrections (see section 5). Provides real-time adaptation to the specific environment, characters, and interaction patterns of this deployment.

Each layer is independent and can be updated on its own schedule.

---

## 3. T2: Interactive Responder

T2 is a mid-size model (7-14B) that handles interactive tasks:

- Generate responses to user input
- Execute multi-step reasoning tasks
- Produce prose updates to state document sections
- Handle tool calls and action dispatch

T2 reads the wirstate (including T1's classifications) as context for its responses. It operates on-demand — triggered by events, not running continuously.

---

## 4. T3: WIR Refiner

T3 is a large model (27-70B+) that runs WIR refinement passes over the state document. This is the model described throughout the WIR State Refinement and Batch Compiler specifications.

T3's role:
- Refine all state variables via micro-prompt batches (n-least-conf)
- Correct T1 classifications that are wrong
- Override T2 responses that are inconsistent with accumulated state
- Maintain long-term coherence across hundreds of interactions
- Compress and reorganise prose state sections

T3 runs less frequently than T1 (per-turn or on a schedule) because it is expensive. But when it runs, it sees the full picture and can correct errors that T1 and T2 made due to their smaller capacity.

---

## 5. Online LoRA Correction

T3's WIR refinement passes produce a continuous stream of implicit quality judgments on T1 and T2 output. When T3 corrects a state variable that T1 classified, or overrides a T2 response via correction tool call, the disagreement constitutes a labelled training example: the input context T1/T2 received, the output it produced, and the output T3 determined was correct.

These corrections drive single-step LoRA updates applied immediately to T1 (and optionally T2). A low-rank adapter (rank 1-4, approximately 1-2M trainable parameters) is attached to the base model weights, which remain frozen. On each disagreement event:

1. The cross-entropy loss is computed on the single incorrect token using T3's correction as the target label.
2. A backward pass computes gradients through the LoRA matrices only — not the base model.
3. A single optimizer step updates the LoRA weights at a very low learning rate (e.g., 1e-5 to 1e-6).

The entire sequence — loss, backward pass, weight update — executes in under 5 milliseconds on a model already resident in VRAM. No batching, no accumulation buffer, no explicit training loop. The correction fires as an event callback and completes before the next T1 classification tick.

Any individual update is negligible — a learning rate of 1e-6 on a single example produces an imperceptible weight change. The value is in accumulation. Over hundreds or thousands of corrections across hours of operation, the LoRA adapter builds up a persistent bias toward the correct output distribution for this specific deployment context. A guard NPC with particular emotional patterns that the general-purpose T1 misclassifies will gradually be classified correctly as the adapter accumulates corrections specific to that character's state dynamics.

Robustness to noisy labels is inherent. T3 is not always correct — some corrections will be wrong. However, if T3's correction accuracy is substantially above chance (e.g., 85-95%), stochastic gradient descent naturally converges toward the majority signal. Incorrect corrections are noise in the gradient; correct corrections are signal. The signal accumulates; the noise averages out. No explicit confidence filtering or validation gate is required, though one could be added as an optimisation.

The LoRA adapter is persistent, deployment-specific state. It serialises alongside the state documents and the paged KV caches as part of the save/checkpoint system. A deployment that has been running for 50 hours has a T1 model that has been continuously adapted to its specific environment, character set, and interaction patterns. Loading a different save or starting a new deployment loads a different LoRA (or no LoRA, starting fresh). The model literally learns the world it operates in.

The correction mechanism composes with the three-layer T1 model strategy: base model capabilities provide general language understanding, offline specialist fine-tuning (section 2.1) provides task-specific classification skill, and online LoRA correction provides deployment-specific adaptation. Each layer is independent and each can be updated on its own schedule — the base model never changes, the specialist is retrained periodically on accumulated logs, and the LoRA evolves continuously during operation.

---

## 6. Interaction Between Tiers

```
Continuous loop:
  T1 reads input → classifies → updates wirstate variables
       │
       ├─ If event detected → triggers T2
       │     T2 reads wirstate → generates response → updates wirstate
       │
       └─ On schedule / per-turn → triggers T3
             T3 reads wirstate → WIR refinement (micro-prompt batch)
                  │
                  ├─ Corrects T1 variables → fires LoRA update on T1
                  ├─ Corrects T2 outputs → fires LoRA update on T2 (optional)
                  └─ Updates wirstate with refined values
                       │
                       └─ Auto-translate → generation context table
                            T2 uses this for next response
```

The tiers form a correction cascade: T1 is fast but inaccurate, T2 is moderate, T3 is slow but authoritative. T3's corrections flow back to T1 via online LoRA, making T1 more accurate over time. Eventually, T1 may become accurate enough for specific variables that T3 refinement is no longer needed for those variables — the system self-optimises.

---

## 7. Layered Cascade Refinement

A single WIR pass refines all variables independently — but many real-world state updates have causal dependencies. An NPC can't react emotionally to an event it hasn't perceived, and it can't perceive an event that hasn't happened. These dependencies form a natural pipeline of layers, where each layer's refined outputs become the next layer's Zone 2b inputs.

### 7.1 The Cascade

Each layer is a full batched micro-prompt WIR pass running at full GPU parallel prefill speed. Between layers, the orchestrator filters and injects:

```
Layer 1: ACTIONS — What happened this tick?
  All entities × action variables → batch refine
  Outputs: action_taken?(Y/N), action_type, action_target, action_location

          │ Filter: extract actions, match to nearby entities
          v

Layer 2: PERCEPTION — Who noticed?
  ALL entities × perception variables → batch refine
  Zone 2b injected: all actions from Layer 1 + physics engine hook data
    (line of sight, distance, sound propagation, occlusion)
  The MODEL decides whether each NPC perceived the event — not the orchestrator.
  Outputs: perceived?(Y/N), perceived_action, perceived_actor

          │ Physics engine hook: inject relationship data, faction standing
          v

Layer 3: EMOTION — How do they feel about it?
  ALL entities × emotion variables → batch refine
  Zone 2b injected: perception results from Layer 2 + relationship data
  NPCs that perceived nothing → emotion unchanged (1 token, no-op)
  Outputs: emotional_response, intensity, valence

          │ Hook: inject world knowledge, quest state
          v

Layer 4: COGNITION — What do they think?
  ALL entities × cognition variables → batch refine
  Zone 2b injected: emotional state from Layer 3 + world knowledge
  NPCs with no new emotion → cognition unchanged (1 token, no-op)
  Outputs: interpretation, intent, plan

          │ Hook: inject available actions, movement constraints
          v

Layer 5: REACTION — What do they do?
  ALL entities × reaction variables → batch refine
  Zone 2b injected: thoughts/plans from Layer 4 + action constraints
  NPCs with no new thoughts → no reaction (1 token, no-op)
  Outputs: reaction_action, reaction_target

          │ Feed back into Layer 1 of the NEXT tick
          v
```

### 7.2 Performance

Each layer runs at full batched prefill throughput. The entity count naturally decreases through the pipeline as filters narrow the set:

| Layer | Description | Entities | Micro-prompts | Time @ 20k tok/s batched |
|---|---|---|---|---|
| 1. Actions | What happened? | 200 (all) | 200 × ~80 tok = 16K | ~800ms |
| 2. Perception | Who noticed? | 200 (all) | 200 × ~80 tok = 16K | ~800ms |
| 3. Emotion | How do they feel? | 200 (all) | 200 × ~80 tok = 16K | ~800ms |
| 4. Cognition | What do they think? | 200 (all) | 200 × ~80 tok = 16K | ~800ms |
| 5. Reaction | What do they do? | 200 (all) | 200 × ~80 tok = 16K | ~800ms |
| **Total** | **5 prefill passes** | | **~80K tokens** | **~4s** |

All NPCs run all layers every tick. NPCs with nothing to process at a given layer produce no-op answers (single token, converges instantly). The batch throughput doesn't degrade for no-op answers — they're still just tokens. The GPU processes them all at the same speed regardless of whether the NPC has something interesting to decide.

At ~4 seconds per tick for 200 entities, the tick rate is fast enough that coordination between NPCs happens naturally across ticks. An NPC drawing a sword (tick N, Layer 5) is perceived by nearby NPCs (tick N+1, Layer 2) within 4 seconds — faster than human reaction time. No scripted coordination needed; emergent behaviour arises from the tick-to-tick feedback loop.

### 7.2.1 Tick Rate and Emergent Coordination

At 1.5-4 seconds per tick, the "delay" between one NPC's action and another's reaction is shorter than real-time human reaction speed. Two guards coordinating a patrol naturally stagger by one tick, which reads as realistic latency — not a bug. A crowd gradually noticing a commotion unfolds over 2-3 ticks (5-10 seconds), which is exactly how real crowds behave.

Cross-entity dependencies resolve themselves across ticks without any explicit coordination logic:

```
Tick N:     NPC_042 draws sword (Layer 5 reaction)
Tick N+1:   NPC_087 sees it (Layer 2 perception, physics hook confirms LOS)
            NPC_087 feels fear (Layer 3 emotion)
            NPC_087 decides to alert guards (Layer 4 cognition)
            NPC_087 shouts for help (Layer 5 reaction)
Tick N+2:   Guard NPCs hear the shout (Layer 2, physics hook: sound propagation)
            Guards feel duty/urgency (Layer 3)
            Guards decide to investigate (Layer 4)
            Guards move toward the location (Layer 5)
```

Three ticks, ~10 seconds of game time, zero scripting. The entire interaction emerges from the cascade pipeline and the physics engine hooks providing ground truth about what each NPC can perceive.

### 7.2.2 Error Cascade as Emergent Behaviour

If Layer 2 misclassifies perception (an NPC "sees" something it shouldn't have), the downstream layers reason from a false premise — the NPC reacts to a phantom event. This is not a bug to be eliminated. It is emergent fallibility.

Real people misperceive situations constantly. A guard who "sees" a shadow and draws his sword, only to realise next tick that nothing is there, is exhibiting realistic cautious behaviour. The T3 LoRA correction mechanism (section 5) ensures these errors become less frequent over time for systematic misclassifications, but occasional errors persist as personality-like traits.

The self-correcting property comes from the tick loop: next tick, the NPC re-evaluates with fresh perception data. If the phantom event isn't confirmed, the emotional and cognitive layers naturally de-escalate. The state variables drift back toward baseline — not because of explicit error handling, but because the model's next-tick reasoning reflects the absence of continuing stimulus.

### 7.2.3 Emergent Temporal Awareness

The tick loop and variable change tracking produce a natural sense of time without any explicit clock mechanism.

Every variable update is a state change. A state change means something happened. The history of state changes across ticks is the NPC's experience of time passing:

- **Was a token updated this tick?** → Something changed. The NPC noticed something, felt something, thought something.
- **What changed in the last tick?** → Immediate context. "The blacksmith just drew his sword."
- **What changed in the last 10 ticks?** → Recent memory. "The marketplace has been tense for the last minute."
- **What changed in the last 100 ticks?** → Medium-term awareness. "There have been several fights today."

The orchestrator maintains a rolling window of Zone 3 change deltas per entity. This history is injected as Zone 2 context for the next tick:

```
[RECENT CHANGES]
tick -1: perceived_action changed (saw sword drawn)
tick -1: emotional_response changed (calm → alarmed)
tick -2: perceived_action changed (heard argument)
tick -5: emotional_response changed (content → uneasy)
tick -30: location changed (arrived at market)
[/RECENT CHANGES]
```

The model reads this as a natural timeline. It doesn't need to know about ticks or seconds — it sees that the NPC recently became alarmed after a period of growing unease, and reasons accordingly. The compression is automatic: old changes that are no longer relevant naturally fall out of the window. Significant changes (emotional shifts, location changes) persist in the summary while routine no-ops are never recorded.

This is the same natural compression described in the WIR State Refinement spec (section 5.1) — the model compresses its own history. The tick-based change log just provides the raw timeline; the model in the cognition layer interprets temporal patterns and durations from the density of changes.

A variable that hasn't changed in 100 ticks signals stability — the NPC's emotional state has been `content` for minutes. A variable that changed 5 times in the last 10 ticks signals turbulence. The model can distinguish "has been angry for a while" from "just became angry" from "keeps fluctuating between angry and calm" — all from the change history alone, with no explicit time modelling.

### 7.2.4 Learned Time Estimation via LoRA Self-Correction

The model can develop an accurate internal clock by predicting elapsed time and being corrected by reality.

Add a time-estimation variable to each NPC's variable set:

```yaml
estimated_elapsed:
  type: byte
  instruction: "How much real time has passed since the last significant
    event this NPC experienced? Estimate in seconds (0-249)."
  bands:
    0-5: "just now"
    6-30: "seconds ago"
    31-120: "a minute or two"
    121-249: "several minutes"
```

Each tick, the model estimates how much time has passed based on the change history and its understanding of the world's pace. The orchestrator knows the actual wall-clock time between ticks (it has a real clock). The actual elapsed seconds are injected as a Zone 2b authoritative value:

```
Micro-prompt for estimated_elapsed:
  Zone 1:  "Estimate seconds since last significant event (0-249):"
  Zone 2:  [RECENT CHANGES] tick -1: saw sword drawn, tick -5: arrived at market
  Zone 2b: actual_elapsed_since_last_event: 8 seconds
  Zone 3:  model's estimate → Ñ (the model guesses)
```

When the model's estimate disagrees with the actual elapsed time in Zone 2b, WIR corrects the Zone 3 answer. This disagreement fires a LoRA update on T1 (section 5): the model predicted 30 seconds but only 8 had passed. The cross-entropy loss on that single token nudges the LoRA weights toward better time estimation.

Over hundreds of ticks, the LoRA accumulates:

- **Early deployment:** The model's time estimates are crude — it has no calibration for how fast this world runs. Errors are large.
- **After 100 ticks (~5-10 minutes):** The LoRA has seen ~100 time estimation corrections. The model starts to develop a sense of the tick rate and the world's pace.
- **After 1000 ticks (~1-2 hours):** The model reliably estimates elapsed time within a few seconds. It has learned that "a fight lasts about 5 ticks" and "walking across the market takes about 10 ticks" from accumulated corrections.
- **After 10000 ticks (~10+ hours):** The model has an accurate internal clock calibrated to this specific deployment's tick rate, world size, and event density.

The time sense is deployment-specific. A deployment with a 2-second tick rate develops a different calibration than one with a 4-second tick rate. A dense urban environment develops a different sense of "how long things take" than a sparse wilderness. The LoRA captures all of this — it is literally the model's learned experience of time in this particular world.

This composes naturally with the cognitive cascade. An NPC in Layer 4 (cognition) that has an accurate sense of elapsed time can reason about urgency differently: "the guards should have arrived by now" vs "the guards were only called a moment ago." The time estimate feeds into planning and expectation, producing NPCs that understand pacing — not because pacing was programmed, but because the model learned it from clock corrections.

The same mechanism generalises to any quantity the model can estimate and the orchestrator can measure:

| Estimated variable | Ground truth source | What the model learns |
|---|---|---|
| Elapsed time | Wall clock | Tick rate, event pacing |
| Distance to event | Physics engine | Spatial scale of the world |
| Crowd density | Entity count query | How busy this area typically is |
| Time of day | Game clock | Diurnal patterns, schedules |
| Temperature | Weather system | Seasonal/environmental patterns |

Each is a prediction-correction loop where the model estimates, reality provides ground truth, and the LoRA accumulates calibration. The model builds an increasingly accurate internal model of the world's physical properties — from time flowing, to distances, to environmental patterns — all through the same single-token LoRA correction mechanism.

### 7.2.5 Predictive Layers — Future Modelling as a General Principle

The prediction-correction loop from section 7.2.4 is not a special case for time estimation. It is the general principle. Every variable in the cascade should have a corresponding **prediction variable** — the model's forecast of what that variable will be next tick. The prediction is scored against reality when the next tick arrives, and the disagreement drives LoRA correction.

#### The Dual-Variable Pattern

For every state variable, add a prediction twin:

```yaml
# State variable (what IS)
emotional_response:
  type: byte
  instruction: "Current emotional state (0=distressed, 249=elated)"

# Prediction variable (what WILL BE)
predicted_emotional_response:
  type: byte
  instruction: "Predicted emotional state NEXT tick (0=distressed, 249=elated)"
```

Each tick, the cascade processes both:

```
Tick N:
  Layer 3 (Emotion):
    emotional_response:           → model evaluates current state → Ñ (176, "elevated")
    predicted_emotional_response: → model predicts next tick      → É (200, "happy")
      (NPC expects the situation to improve — help is coming)

Tick N+1:
  Layer 3 (Emotion):
    Zone 2b injection: previous_predicted_emotional_response: É (200)
    Zone 2b injection: actual from this tick's evaluation:    µ (180, slightly better)

    The prediction was É (200) but reality was µ (180).
    Disagreement → LoRA correction fires.
    The model learns: "situations like this improve, but not as fast as I expected."
```

The prediction variable is in Zone 3 (mutable, WIR-refined). The previous tick's prediction is injected as Zone 2b (authoritative ground truth) in the next tick. The model sees its own past prediction alongside what actually happened, and WIR corrects the prediction variable if it would make the same error again.

#### Extended Cascade with Prediction Layers

The 5-layer cascade gains a 6th layer — prediction — that runs after reaction:

```
Layer 1: ACTIONS        — What happened this tick?
Layer 2: PERCEPTION     — Who noticed?
Layer 3: EMOTION        — How do they feel?
Layer 4: COGNITION      — What do they think?
Layer 5: REACTION       — What do they do?
Layer 6: PREDICTION     — What do they expect to happen next?
```

Layer 6 predicts the next tick's Layer 1-5 values. When the next tick arrives, the actual Layer 1-5 values are compared to Layer 6's predictions. Every disagreement is a LoRA training signal.

```
Layer 6 micro-prompts (per NPC, per predictable variable):

  predicted_nearby_action:    "Will someone nearby act? What?"
  predicted_perception:       "Will I notice something new?"
  predicted_emotion:          "How will I feel next tick?"
  predicted_cognition:        "Will my plans change?"
  predicted_own_action:       "What will I do next?"
  predicted_other_reaction:   "How will others react to what I just did?"
```

#### What the Model Learns

Over thousands of ticks, the LoRA accumulates a model of the world's dynamics — not just static properties but causal patterns:

| Prediction type | What accumulates in LoRA | Emergent capability |
|---|---|---|
| Emotional trajectory | "Anger fades after ~10 ticks without stimulus" | Emotional pacing, realistic cool-down |
| Other NPC reactions | "Guards respond to shouts within 2-3 ticks" | Anticipation, planning around expected responses |
| Own action outcomes | "Starting a fight leads to guards arriving" | Consequential thinking, risk assessment |
| Environmental patterns | "The market gets busy around tick 500" | Diurnal awareness, routine behaviour |
| Social dynamics | "This NPC holds grudges for ~100 ticks" | Relationship modelling, social intelligence |

The NPC is not following a behaviour tree or a scripted emotional arc. It is making predictions about the future, observing what actually happens, and learning from the delta. An NPC that has been in 50 fights learns that fights escalate in a specific pattern. An NPC that has watched the market for 1000 ticks learns when the busy periods are. An NPC that has interacted with a specific other NPC 200 times learns that character's patterns.

#### Prediction Confidence as a Decision Signal

The model's confidence in its predictions (the entropy of the Zone 3 logits during the prediction layer) is itself a useful signal:

- **High confidence prediction:** The NPC "knows" what will happen. It can plan ahead. A guard who confidently predicts "the thief will run toward the east gate" can preemptively move to intercept.
- **Low confidence prediction:** The NPC is uncertain. It may act cautiously, gather more information, or wait. An NPC that can't predict whether a stranger is friendly or hostile will be wary.
- **Confident prediction that turns out wrong:** Surprise. The NPC expected one thing and got another. This can trigger emotional responses (shock, delight, confusion) in the next tick's Layer 3, producing realistic surprise reactions.

The prediction confidence doesn't need a special mechanism — it falls out naturally from the WIR refinement. The n-least-conf scoring already measures how confident the model is in each answer token. A prediction variable where the model assigns low probability to its own answer is a variable where the NPC is genuinely uncertain about the future.

#### Cost

Layer 6 adds one more prefill pass per tick. For 200 NPCs with ~10 prediction variables each:

| Component | Cost |
|---|---|
| Layer 6 prefill | 200 × ~80 tok = 16K tokens → ~800ms @ 20k tok/s |
| LoRA corrections (comparing predictions to actuals) | ~100 disagreements × 5ms each = ~500ms |
| **Total addition per tick** | **~1.3s** |

Total cascade with prediction: 6 layers × ~800ms + ~500ms LoRA = **~5.3s per tick**. Still under 6 seconds for a complete cognitive loop with future modelling across 200 entities.

The LoRA corrections from prediction errors are the highest-value training signal in the system. They capture causal dynamics — "when X happens, Y follows" — rather than just classification accuracy. A model that learns to predict well has effectively learned a causal model of its world.

### 7.3 Per-NPC Stacks

Rather than a single global cascade with spatial filtering between layers, each NPC runs its own independent 5-layer cascade. Every NPC processes all five layers every tick. The question "did anything happen near me?" is asked by the model inside Layer 2's micro-prompt, not by the orchestrator with a radius cutoff.

This is cleaner for two reasons:

1. **No orchestrator-side spatial logic.** The model is better at fuzzy proximity reasoning ("close enough to hear the shout", "could see through the window") than a hard geometric radius. The model makes the perception decision; the orchestrator just provides the raw facts.

2. **Simpler pipeline.** Every NPC enters every layer. No filter step, no subset tracking, no entity-set management between layers. The orchestrator's job is: run Layer 1 for all NPCs → inject outputs into Layer 2 for all NPCs → run Layer 2 → inject → run Layer 3 → etc.

NPCs that have nothing to perceive in Layer 2 simply output `perceived?: N` and their Layers 3-5 produce no-op values (emotion: unchanged, cognition: no new thoughts, reaction: none). These are single-token answers that converge in one pass — negligible cost.

### 7.4 Physics Engine Hooks

The orchestrator does not need to understand spatial relationships, line of sight, sound propagation, or any other physical simulation. That's what the physics engine is for. External hooks at any cascade layer can inject physics engine outputs as Zone 2b variables:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Layer 1     │────>│  Physics     │────>│  Layer 2     │
│  Actions     │     │  Engine Hook │     │  Perception  │
│  (WIR pass)  │     │              │     │  (WIR pass)  │
│              │     │  Queries:    │     │              │
│  Output:     │     │  - LOS check │     │  Zone 2b:    │
│  NPC_042     │     │  - Distance  │     │  can_see: Y  │
│  drew_sword  │     │  - Occlusion │     │  distance: 12│
│              │     │  - Sound     │     │  heard: Y    │
│              │     │    propagation│     │  sound_level:│
│              │     │              │     │  loud        │
└──────────────┘     └──────────────┘     └──────────────┘
```

The hook fires between Layer 1 and Layer 2. For each NPC, it queries the physics engine: "given NPC_042 drew a sword at position X, can NPC_087 see it? hear it? how far away?" The answers are injected as Zone 2b authoritative values. The model in Layer 2 then decides *whether the NPC noticed* given that it *could* see/hear — a sleeping NPC with `can_see: Y` might still output `perceived?: N`.

Hooks can inject at any layer boundary:

| Hook point | What it provides | Source |
|---|---|---|
| Before Layer 2 (Perception) | Line of sight, distance, sound propagation, occlusion | Physics/raycast engine |
| Before Layer 3 (Emotion) | Relationship data, faction standing, history | Game state DB |
| Before Layer 4 (Cognition) | World knowledge, quest state, inventory | Game logic |
| Before Layer 5 (Reaction) | Available actions, movement constraints, cooldowns | Animation/pathfinding system |

The model handles the fuzzy reasoning (does this NPC care? how do they interpret it?). The physics engine handles the hard constraints (can they physically see it?). Clean separation.

### 7.5 Layer-to-Layer Injection

Between layers, the orchestrator performs a lightweight inject step:

1. **Read** the refined Zone 3 values from the completed layer.
2. **Run hooks** — external systems (physics engine, game state DB) provide authoritative data for the next layer.
3. **Inject** previous layer outputs + hook data as Zone 2b values in the next layer's micro-prompts.

This is the same hook mechanism described in the WIR State Refinement spec (section 3.5), operating between cascade layers within a single tick. Each hook point is a standard pre-refine hook that has read/write access to the wirstate between layers.

### 7.4 Layer Definitions

Layers are defined in the wirstate configuration, not hardcoded. Different applications define different pipelines:

```yaml
# Game NPC cognitive pipeline — per-NPC stacks, physics hooks
cascade:
  - name: actions
    variables: [action_taken, action_type, action_target, action_location]
    entities: all
    inject_from: previous_tick.reaction  # feedback loop

  - name: perception
    variables: [perceived, perceived_action, perceived_actor]
    entities: all  # every NPC evaluates; model decides who noticed
    inject_from: actions
    hooks:
      - physics_engine:
          provides: [can_see, distance, heard, sound_level, line_of_sight]
          queries: raycast, sound_propagation, occlusion

  - name: emotion
    variables: [emotional_response, intensity, valence]
    entities: all  # NPCs with no perception → emotion unchanged (no-op)
    inject_from: perception
    hooks:
      - game_state:
          provides: [relationship, faction_standing, history_with_actor]

  - name: cognition
    variables: [interpretation, intent, plan]
    entities: all
    inject_from: emotion
    hooks:
      - game_state:
          provides: [quest_state, world_knowledge, inventory]

  - name: reaction
    variables: [reaction_action, reaction_target]
    entities: all
    inject_from: cognition
    feedback_to: actions  # outputs feed into Layer 1 of next tick
    hooks:
      - animation_system:
          provides: [available_actions, movement_constraints, cooldowns]
      - pathfinding:
          provides: [reachable, path_cost, blocked]
```

```yaml
# Monitoring agent pipeline (different domain, same mechanism)
cascade:
  - name: sensor_read
    variables: [cpu_load, mem_pressure, disk_io, net_latency]
    entities: all_hosts
    hooks:
      - metrics_api:
          provides: [raw_cpu, raw_mem, raw_disk, raw_net]

  - name: anomaly_detect
    variables: [is_anomalous, anomaly_type, severity]
    entities: all_hosts
    inject_from: sensor_read

  - name: root_cause
    variables: [likely_cause, affected_services, blast_radius]
    entities: all_hosts  # all hosts evaluate; model decides relevance
    inject_from: anomaly_detect
    hooks:
      - topology_db:
          provides: [upstream_services, downstream_services, dependency_graph]

  - name: response
    variables: [recommended_action, urgency, escalate]
    entities: all_hosts
    inject_from: root_cause
    hooks:
      - runbook_api:
          provides: [available_remediations, escalation_policy]
```

### 7.5 Interaction with Online LoRA

T3 refinement (section 5) can run over any individual layer's outputs. If T3 determines that Layer 2 perception classifications are frequently wrong for a specific entity type, the LoRA corrections accumulate on the variables in that layer. Over time, Layer 2 becomes more accurate for that entity type, which cascades into more accurate emotion, cognition, and reaction layers — without T3 needing to correct every downstream layer individually. Fix the perception, and the downstream reasoning improves for free.

### 7.6 Single-Pass vs Multi-Pass Within Layers

Each layer in the cascade is itself a single WIR pass (one prefill over the micro-prompt batch). For most variables, single-pass convergence is sufficient — the instruction is clear, the context is fresh from the previous layer, and the answer is a single token.

However, a layer can optionally run 2-3 WIR passes internally if its variables have complex interdependencies. For example, if Layer 3 (emotion) has both `emotional_response` and `intensity` variables that depend on each other, running 2 passes allows them to reach mutual consistency. The orchestrator detects convergence (no answer tokens changed) and advances to the next layer.

The cascade structure (Layer 1 → 2 → 3 → 4 → 5) handles inter-variable dependencies that cross conceptual boundaries. Intra-layer WIR passes handle dependencies within a conceptual boundary. Together, they resolve arbitrary dependency chains in a small number of fast prefill passes.

---

## 8. Pipeline Self-Optimisation

The LoRA correction mechanism (section 5) trains the model to be better at evaluating existing variables. This section describes a separate, faster training loop that optimises **the pipeline itself** — which variables exist, how they're layered, what gates what, and whether a variable is pulling its weight.

This is not gradient descent. It is experimental pruning and mutation of the cascade configuration, evaluated by measuring impact on task success and prediction accuracy. The loop runs at the speed of config file edits, not weight updates.

### 8.1 The Only Honest Test

Simple heuristics about variable behaviour are misleading:

- A variable that **never changes** is not necessarily dead. It may be foundational — a deeply held belief, a physical constant, a core identity trait. "Is this NPC loyal to the king?" might not change for 10,000 ticks and be the single most important variable in the stack.
- A variable that **changes every tick** is not necessarily too volatile. It may be correctly tracking something dynamic — a combat target's position, a rapidly evolving conversation tone.
- A variable with **low prediction accuracy** is not necessarily useless. It may be tracking something genuinely unpredictable that still provides useful context ("I don't know what the player will do next, but my uncertainty about it informs my caution").

Change rate, entropy, and correlation statistics are useful signals for *identifying candidates* to test, but they cannot determine whether a variable is useful. The only honest test is experimental: **remove the variable, measure whether task success and downstream prediction accuracy degrade.** If they don't, the variable wasn't contributing. If they do, put it back.

### 8.2 Experimental Pruning

The optimisation loop runs periodically (every K ticks or on a schedule):

```
Pipeline Optimisation Cycle:

1. MEASURE BASELINE
   Record prediction accuracy per variable per layer, task success
   metrics, and overall cascade coherence over a window of N ticks.

2. IDENTIFY CANDIDATES
   Flag variables for experimental pruning based on statistical signals:
   - Stable variables (low change rate): candidates for demotion
     to a lower-frequency evaluation tier (not removal)
   - High-entropy variables: candidates for splitting into
     more specific sub-variables
   - Correlated variable pairs: candidates for merging
   - Variables with no measurable downstream influence:
     candidates for removal

3. RUN EXPERIMENT
   For each candidate, fork the pipeline:
   - Control: current pipeline, unchanged
   - Experiment: pipeline with the candidate modification
   Run both for N ticks on the same input stream.

4. EVALUATE
   Compare prediction accuracy and task success between
   control and experiment. Statistical significance required
   (not just "it seemed fine").

5. COMMIT OR REVERT
   If the experiment performs equal or better: keep the change.
   If worse: revert. Log the result either way — a variable
   that survives a pruning attempt is confirmed important.
```

### 8.3 Variable Tiering (Not Just Keep/Remove)

Pruning is too binary. A variable that rarely changes shouldn't be removed — it should be **demoted** to a lower-frequency evaluation tier. This creates a hierarchy of evaluation frequency matched to how dynamic each variable actually is:

```
Tier 0: EVERY TICK         — fast-changing, operationally critical
  combat_target, perceived_action, emotional_response
  Full micro-prompt evaluation every tick.

Tier 1: EVERY 10 TICKS     — moderately dynamic
  current_goal, relationship_with_nearby, mood_baseline
  Evaluated periodically. Cached value used between evaluations.

Tier 2: EVERY 100 TICKS    — slow-moving state
  core_beliefs, faction_loyalty, long_term_plan
  Evaluated rarely. Almost always returns the same value.
  Still contributes to context (injected as Zone 2b for other variables).

Tier 3: ON-DEMAND          — foundational, changes only on major events
  identity, species, fundamental_values
  Only re-evaluated when a major state change is detected
  in a higher tier (e.g., a Tier 0 variable crosses a threshold).
```

The optimisation loop manages tier assignment. A variable that hasn't changed in 100 ticks gets demoted from Tier 0 to Tier 1. A Tier 2 variable that suddenly changes gets promoted back to Tier 0. The tiers are dynamic, not fixed.

**Compute savings:** If 60% of variables are Tier 1-3 at any given time, the per-tick micro-prompt batch shrinks by 60%. Only the Tier 0 variables (the ones that are actually changing) consume prefill compute every tick. The rest contribute their cached values as context without being re-evaluated.

### 8.4 Protected Variables — Modelling Accuracy vs Task Success

Some variables don't directly drive task success but are essential for the model's internal reasoning coherence. These should be protected from pruning even if experimental removal shows no measurable drop in task metrics.

**Emotions** are the canonical example. An NPC doesn't need to track `emotional_response` to walk to the market. Removing it might not degrade task completion rates at all. But an NPC without emotional state:
- Makes worse predictions about other NPCs' behaviour (can't model that the angry guard will be aggressive)
- Produces incoherent action sequences (calm → attack → calm, with no emotional arc)
- Loses the emergent personality that makes it interesting
- Degrades the quality of T2's generated dialogue (no emotional context to draw from)

These effects are real but hard to capture in a simple accuracy metric. The experimental pruning system (section 8.2) would see "task success unchanged" and remove the variable, producing a technically functional but qualitatively worse system.

**Solution: variable tags that control pruning eligibility.**

```yaml
emotional_response:
  type: byte
  protected: true        # never auto-pruned
  reason: modelling       # exists for reasoning coherence, not task success
  tier_eligible: true     # can still be demoted to lower evaluation frequency

combat_target:
  type: string
  protected: false        # can be pruned if experiments show no impact

faction_loyalty:
  type: byte
  protected: true
  reason: identity        # core to NPC identity, removal changes who they are
  tier_eligible: true
```

Protected variables can still be **tiered** (evaluated less frequently if stable) but never **removed** by the optimisation loop. The human designer marks which variables are load-bearing for coherence at design time. The model can suggest protecting a variable via the `meta_pipeline_suggestion` mechanism if it notices that removing something degraded its reasoning quality even when metrics didn't capture it.

More broadly, there are three categories:

| Category | Prunable? | Tierable? | Examples |
|---|---|---|---|
| **Operational** — directly drives actions | Yes (if experiment shows no impact) | Yes | combat_target, path_destination |
| **Modelling** — supports reasoning quality | No (protected) | Yes | emotional_response, relationship, mood |
| **Identity** — defines who the entity is | No (protected) | Tier 3 only (rarely re-evaluated) | faction_loyalty, core_values, personality |

The optimisation loop only experiments with operational variables. Modelling and identity variables are maintained by design, with the human (or T3) deciding what belongs in each category.

### 8.5 Model-Proposed Mutations

The LLM itself can propose pipeline changes. Add a self-analysis variable to the prediction layer:

```yaml
meta_pipeline_suggestion:
  type: string
  layer: prediction
  instruction: "Given your current variables and your recent prediction
    errors, suggest ONE change to improve your predictions. Options:
    ADD(name, description), REMOVE(name), SPLIT(name, into what),
    MERGE(name1, name2), REORDER(name, move to layer N).
    If no change needed, output NONE."
```

Every N ticks, the model looks at its own prediction failures and proposes a structural change:

- "ADD(time_of_day_awareness, whether shops are open or closed) — I keep being surprised by merchant availability"
- "SPLIT(emotional_response, into anger and fear) — I'm conflating two different reactions"
- "MERGE(perceived_action and perceived_actor, into perceived_event) — these always change together"
- "REMOVE(wind_direction) — this has never influenced any of my decisions"
- "REORDER(intent, move to layer 2) — I need to know my intent before evaluating emotion"

The optimisation loop takes these suggestions as candidate mutations and runs the same experimental evaluation (section 8.2). The model proposes; the experiment validates. Bad suggestions are discarded with no harm. Good suggestions improve the pipeline.

#### Self-Authored Variables and Hooks

The model doesn't just propose that a variable should exist — it creates the full wiring. When T2 or T3 determines it needs to stay aware of something, it can:

1. **Define the variable** — name, type, instruction text, layer placement, bands for auto-translation.
2. **Write the hook** — a script that feeds the variable with data from external systems.
3. **Set the protection level** — whether the variable is operational (prunable) or modelling (protected).

The model outputs these as structured tool calls during a T2 decode or T3 refinement pass:

```json
{
  "action": "create_variable",
  "variable": {
    "name": "merchant_schedule_phase",
    "type": "byte",
    "layer": "perception",
    "protected": true,
    "reason": "modelling",
    "instruction": "Current phase of the nearest merchant's daily schedule (0=closed/sleeping, 60=opening, 120=peak trade, 180=winding down, 249=closing)",
    "bands": {
      "0-50": "closed",
      "51-90": "opening up",
      "91-160": "busy trading",
      "161-210": "winding down",
      "211-249": "closing for the night"
    }
  },
  "hook": {
    "point": "before_perception",
    "type": "game_state_query",
    "script": "query_merchant_schedule.py",
    "code": "#!/usr/bin/env python3\nimport sys, json\nstate = json.load(open(sys.argv[1]))\n# Query game clock, find nearest merchant, look up schedule\ngame_hour = state['external']['game_clock_hour']\nphase = int((game_hour / 24.0) * 249)\nstate['zone2b']['merchant_schedule_phase'] = phase\njson.dump(state, open(sys.argv[1], 'w'))\n"
  }
}
```

The orchestrator receives this, validates the hook (sandboxed execution, no network access, no filesystem writes outside the state directory), and installs it into the pipeline. The variable is live on the next tick. The experimental evaluation loop (section 8.2) monitors whether the new variable actually improves predictions. If the model added it as `protected: true` (modelling category), it won't be auto-pruned, but the evaluation still logs its contribution so humans can review.

**What this means in practice:**

An NPC that keeps getting surprised by shop closures doesn't just adapt its predictions (LoRA) — it creates a variable to track merchant schedules, writes a hook that queries the game clock, and installs the whole thing into its own perception layer. Next tick, it knows when shops open and close. It bootstrapped a new sensory input because it noticed it needed one.

The same mechanism extends to any awareness gap:

| Model notices... | It creates... | Hook queries... |
|---|---|---|
| Keeps getting surprised by weather changes | `weather_trend` variable | Weather system API |
| Can't predict when guards change shift | `guard_shift_phase` variable | Guard schedule DB |
| Doesn't know why some NPCs are hostile | `local_reputation` variable | Faction/reputation system |
| Loses track of conversations mid-way | `conversation_turn_count` variable | Dialogue state tracker |
| Can't estimate distances accurately | `distance_to_target` variable | Physics engine raycast |

Each time, the model identifies the gap, creates the variable with appropriate layer placement and bands, writes a hook to feed it, and the pipeline absorbs it. The system grows new senses.

**Safety constraints on self-authored hooks:**

- Hooks execute in a sandbox — no network, no arbitrary file access, read/write only to state files.
- Hook code is validated by the orchestrator before installation (syntax check, sandboxing rules).
- A maximum hook count prevents unbounded growth (e.g., max 50 hooks per entity).
- T3 periodically reviews model-created hooks and can disable or modify them.
- All hook creation events are logged for human review. The model can create cognitive architecture but a human can always audit and override it.

Over time, the pipeline evolves:
- **Hour 1:** Human-designed initial stack. 20 variables, 5 layers, 4 hooks. Reasonable but generic.
- **Hour 10:** Model has added 5 variables with self-authored hooks for things it noticed it needed. Optimisation has pruned 2 useless original variables, demoted 8 to lower tiers. Pipeline is leaner and more aware.
- **Hour 100:** The pipeline has been substantially restructured. Variables the human designer didn't think of are now core to the NPC's cognition. The model has written hooks into game systems the designer didn't anticipate connecting. Layer ordering has been adjusted. Tier assignments have stabilised. The pipeline is now deployment-specific — optimised for *this* world with *these* NPCs and *these* interaction patterns. And the model built most of it itself.

### 8.6 Three Learning Loops

The system now has three distinct learning mechanisms operating at different speeds:

| Loop | What it optimises | Speed | Mechanism |
|---|---|---|---|
| **LoRA correction** (section 5) | Model weights — how well the model evaluates existing variables | ~5ms per correction, accumulates over hours | Gradient descent on single-token disagreements |
| **Pipeline optimisation** (this section) | Cascade structure — which variables exist, their layering and tiering | Every K ticks (~minutes), experiments over N ticks | Experimental mutation with statistical evaluation |
| **Prediction-correction** (section 7.2.5) | World model — the model's understanding of dynamics, timing, causality | Every tick, correction via Zone 2b injection | Prediction vs reality, fed back as LoRA training signal |

Each loop is independent. Each operates on a different timescale. Together they produce a system that:
1. Gets better at evaluating state (LoRA)
2. Evolves the right state to evaluate (pipeline optimisation)
3. Builds an accurate model of how state evolves (prediction)

The base model provides general intelligence. The LoRA provides deployment-specific skill. The pipeline provides deployment-specific structure. The prediction loop provides deployment-specific world knowledge. All four layers compose, and all but the base model evolve continuously during operation.

---

## 9. Visual Imagination — Diffusion Integration

The cascade pipeline produces structured internal state at every layer: what the NPC perceives, how it feels, what it thinks, what it expects to happen. This state is text — but text is exactly what diffusion models consume as conditioning. The NPC's internal state can be routed to a diffusion engine to produce visual representations of what the NPC is imagining, remembering, or expecting.

### 9.1 The Insight

Layer 5 (Prediction) already produces structured expectations about the future:

```
Layer 2 output:  perceived: Y, perceived_action: "man drawing sword", perceived_location: "near fountain"
Layer 3 output:  emotional_response: "afraid", intensity: 200
Layer 4 output:  interpretation: "imminent attack on merchant", plan: "shout for guards"
Layer 5 output:  predicted_nearby_action: "fight breaks out at fountain"
```

This is a scene description. Not a carefully crafted prompt, but a raw, emotionally-coloured, subjectively-biased description of what the NPC thinks is about to happen. Feed it to a diffusion model and you get the NPC's mental image — not an objective rendering of the scene, but how *this NPC* imagines it, coloured by its emotional state and biases.

### 9.2 Cascade-to-Conditioning Pipeline

A new output hook after each cascade tick assembles state variables into diffusion conditioning:

```
┌──────────────────────────────────────────────────────────────┐
│              Cascade Layers 1-6 (per tick)                    │
│                                                              │
│  Perception → Emotion → Cognition → Reaction → Prediction   │
│       │           │          │           │          │         │
│       └───────────┴──────────┴───────────┴──────────┘         │
│                           │                                   │
│              Assembled state variables                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           v
              ┌────────────────────────┐
              │  Imagination Assembler  │
              │                        │
              │  Composes state into:  │
              │  - Scene description   │
              │  - Emotional colouring │
              │  - Attention focus     │
              │  - Temporal context    │
              │  - Uncertainty/blur    │
              └───────────┬────────────┘
                          │
                          v
              ┌────────────────────────┐
              │  Diffusion Engine      │
              │  (Stable Diffusion,    │
              │   Flux, etc.)          │
              │                        │
              │  Conditioning:         │
              │  - Text prompt         │
              │  - Strength/weight     │
              │  - Style modifiers     │
              └───────────┬────────────┘
                          │
                          v
              ┌────────────────────────┐
              │  NPC's Mental Image    │
              │  (subjective, biased,  │
              │   emotionally coloured)│
              └────────────────────────┘
```

### 9.3 Emotional Colouring

The NPC's emotional state modifies the diffusion conditioning, producing images that reflect subjective experience rather than objective reality:

| Emotional state | Diffusion modifier | Visual effect |
|---|---|---|
| Afraid (intensity high) | Darker palette, high contrast, vignetting | Threatening, shadows emphasised |
| Angry (intensity high) | Warm/red shift, sharp edges, high saturation | Intense, aggressive framing |
| Content (intensity low) | Warm light, soft focus, golden hour palette | Pleasant, nostalgic quality |
| Confused | Blur, overlapping elements, unstable composition | Dreamlike, uncertain |
| Alert/focused | Sharp detail on attention target, peripheral blur | Spotlight effect, tunnel vision |
| Nostalgic/remembering | Desaturated, grain, soft edges, past-tense styling | Memory quality, faded |

These modifiers are applied as style prompts, ControlNet conditioning, or post-processing adjustments. The diffusion model doesn't need to understand emotions — it just receives modified prompts and parameters that produce the right visual quality.

### 9.4 Three Modes of Visual Imagination

The cascade state supports three distinct visual modes:

**Perception image — "What I see right now"**
Built from Layer 2 (Perception) state. The NPC's subjective view of the current scene, coloured by attention and emotion. Not a camera render — a *reconstruction* of what the NPC noticed, with unattended details missing or vague.

```
Prompt assembled from Layer 2 + Layer 3:
"A man drawing a sword near a stone fountain in a crowded marketplace,
 threatening, seen from 20 meters away, partially obscured by market
 stalls, {afraid, high contrast, dark shadows, the man is the focal
 point, background crowds are blurred}"
```

**Prediction image — "What I think will happen"**
Built from Layer 5 (Prediction) state. The NPC's mental image of the expected near future. Inherently uncertain — the diffusion strength can map to prediction confidence, producing sharper images for confident predictions and dreamlike/blurry images for uncertain ones.

```
Prompt assembled from Layer 5 + confidence:
"A fight breaking out at a stone fountain, a merchant being attacked,
 guards running toward the scene, {anticipated, slightly blurred
 (confidence: 60%), warm afternoon light, motion blur on figures}"
```

**Memory image — "What I remember"**
Built from the change history (section 7.2.3). Older memories are assembled from compressed state with lower detail. Recent memories are vivid. The temporal decay of the change window naturally maps to diffusion parameters: recent events are sharp and saturated, old events are faded and vague.

```
Prompt assembled from tick history (100 ticks ago):
"A quiet morning at the market, the blacksmith opening his shop,
 sunlight on cobblestones, {remembered, desaturated, soft grain,
 nostalgic warmth, low detail on faces}"
```

### 9.5 Practical Considerations

**Cost:** Diffusion image generation is expensive (2-10 seconds per image depending on model and resolution). This is not run every tick. It runs on demand — when the game system needs a visual representation of an NPC's inner state (dream sequences, flashback cutscenes, UI showing "what the NPC is thinking"), or for debugging/visualisation during development.

**Resolution:** The cascade state produces *compositional* descriptions, not pixel-level detail. The diffusion model fills in the visual detail. This is a feature: the NPC's "imagination" has the right level of abstraction. It pictures "a fight at the fountain" — it doesn't render each cobblestone. Just like human mental imagery.

**Consistency:** Repeated imagination renders for the same NPC should have visual consistency — same style, same colour palette tendencies, same level of detail. This can be achieved with per-NPC LoRA adapters on the diffusion model (separate from the T1 inference LoRA), or with fixed seed + consistent style prompts derived from the NPC's identity variables.

**Integration with self-authored hooks (section 8.5):** A model that creates its own perception variables might also create variables specifically to improve its imagination quality. An NPC that has developed `weather_trend` and `time_of_day_awareness` variables produces more accurate and atmospherically consistent mental images because those variables feed into the imagination assembler's prompt.

### 9.6 Beyond NPCs — General Visual State Rendering

The same mechanism applies to any cascade pipeline, not just game NPCs:

- **Monitoring agent:** Imagining what the system architecture looks like under stress — a diffusion-rendered "mental model" of server load, with overloaded nodes glowing red and healthy nodes green.
- **Research agent:** Visualising a hypothesis — "what I think the protein structure looks like given the data so far."
- **Creative assistant:** Imagining the scene being written — the cascade state from a story's characters produces illustrations that reflect the narrative's current emotional tone.

The cascade state is a rich, structured, emotionally-coloured description of an internal model. Diffusion is a general-purpose structured-text-to-image engine. The combination is a general-purpose imagination system.

### 9.7 Token Framebuffer — LLM-Rendered Seed Images

Rather than relying solely on text prompts to condition the diffusion model, the LLM can directly render a crude spatial image as a grid of tokens — a framebuffer made of text. This image captures the model's spatial reasoning (composition, lighting, colour relationships, object placement) and is used as the init/seed image for the diffusion step, which adds photorealistic detail while preserving the model's intended layout.

#### 9.7.1 The Format

A markdown table (or similar grid) where each cell is a "pixel" encoded as two characters — one for intensity, one for colour:

```
32×32 token framebuffer (1024 cells, 2 tokens each):

|D3|E3|E3|F2|F2|G2|G1|H1|H1|I1|...|
|D3|D4|E4|E3|F3|F2|G2|G1|H1|I1|...|
|C4|D4|D4|E4|E3|F3|F2|G2|G1|H1|...|
|B5|C5|C4|D4|E4|E3|F3|F2|G2|G1|...|
|A6|B6|B5|C5|D4|D4|E3|F3|F2|G2|...|
|...(32 rows)                       |

Intensity: A=darkest ... Z=brightest (26 levels)
Colour:    0=black 1=red 2=orange 3=yellow 4=green
           5=cyan 6=blue 7=purple 8=white 9=grey
```

The example above shows a gradient from dark-green (bottom-left) to bright-red (top-right) — perhaps a sunset over a hillside as imagined by the NPC.

#### 9.7.2 Rendering as WIR Micro-Prompts

Each cell in the framebuffer is an independent micro-prompt, refined in parallel:

```
Micro-prompt for pixel (row=12, col=7):

Zone 1 (instruction, cached forever):
  "You are rendering your mental image of the current scene.
   Output the visual content at row 12, column 7 of a 32×32 grid.
   First character: intensity (A=dark, Z=bright).
   Second character: colour palette index (0-9).
   The grid covers your full visual field, row 0 = top, row 31 = bottom,
   col 0 = left, col 31 = right."

Zone 2 (scene context, cached per-turn):
  "Scene: afternoon marketplace, stone fountain center-right,
   a man with a drawn sword near the fountain, merchant cowering behind
   a stall to the left, crowd scattering, warm sunlight from upper-left,
   long shadows stretching right."

Zone 2b (spatial reference, from physics/render engine):
  "Row 12, col 7 corresponds to: mid-height, left-of-center.
   Objects at this position: edge of a market stall, wooden frame.
   Lighting: partial shadow from awning above.
   Camera reference region colour: brown/dark."

Zone 3 (answer — 2 tokens):
  C3
  (C=dim, 3=yellow-brown — the shadowed wooden stall frame)
```

**All 1024 cells refine in parallel** in a single WIR pass. Camera/engine reference images can be decomposed into Zone 2b data per cell — the render engine provides ground-truth colour/intensity for each grid position, and the model's job is to render its *subjective version* of that scene, modified by attention, emotion, and imagination.

At 20k tok/s batched: 1024 micro-prompts × ~80 tokens = ~82K tokens → **~4 seconds** for a complete 32×32 mental image.

#### 9.7.3 Reference Image Integration

The token framebuffer doesn't need to be imagined from scratch. External reference images — from the game camera, a render engine, or a previous frame — can be decomposed into per-cell Zone 2b data:

```
Reference pipeline:

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Game camera  │────>│  Decomposer  │────>│  Per-cell    │
  │  or render   │     │              │     │  Zone 2b     │
  │  engine      │     │  32×32 grid  │     │  injection   │
  │  screenshot  │     │  sample:     │     │              │
  │              │     │  avg colour, │     │  "region is  │
  │              │     │  brightness, │     │   brown,     │
  │              │     │  dominant    │     │   dim,       │
  │              │     │  object      │     │   wooden     │
  │              │     │              │     │   stall"     │
  └──────────────┘     └──────────────┘     └──────────────┘
```

The model receives the objective reference ("this region is brown, it's a wooden stall") and produces its subjective rendition. A frightened NPC might render the same stall darker with more contrast. A content NPC renders it warmer. An NPC that's not paying attention to that area might render it as vague/neutral. The reference grounds the composition; the model's state colours the rendering.

The reference image can also be an ASCII or SVG representation generated by the model itself in a prior pass — a rough compositional sketch that gets refined into the token framebuffer, which then seeds the diffusion model. Multiple levels of refinement, each adding detail:

```
Level 1: Model generates ASCII sketch (T2 decode, ~50 tokens)
  "  ~~~sky~~~  "
  " /mountain\  "
  " |fountain|  "
  " --market--  "

Level 2: Token framebuffer (WIR parallel, 32×32, ~4s)
  Spatial detail, colour, lighting, emotional colouring

Level 3: Diffusion model (img2img from framebuffer, 2-10s)
  Photorealistic detail, texture, coherent rendering
```

#### 9.7.4 Resolution and Scaling

| Grid size | Cells | Tokens | Time @ 20k tok/s | Use case |
|---|---|---|---|---|
| 16×16 | 256 | ~20K | ~1s | Quick compositional sketch |
| 32×32 | 1024 | ~82K | ~4s | Standard mental image |
| 64×64 | 4096 | ~328K | ~16s | Detailed scene (needs chunking) |

16×16 is enough for composition and colour blocking. 32×32 adds spatial detail. Beyond 64×64 the token count exceeds practical batch sizes and the diffusion model doesn't benefit from finer init images at low denoising strength anyway.

The sweet spot is probably 32×32 for scene imagination and 16×16 for quick "what am I picturing" checks during the cognitive cascade.

#### 9.7.5 What This Enables

The combination of LLM spatial reasoning + diffusion photorealism produces capabilities that neither system has alone:

- **NPCs that can draw:** An NPC artist in a game could produce images of what it's imagining. The token framebuffer captures its artistic intent; the diffusion model renders it.
- **Dream sequences:** An NPC's memory state assembled as a framebuffer produces surreal, emotionally-coloured imagery with the narrative coherence of the LLM and the visual quality of diffusion.
- **Debugging tool:** Visualising what the NPC "thinks" the scene looks like reveals perception errors, attention blind spots, and emotional biases in a way that reading variable values never could.
- **Witness testimony:** An NPC that witnessed an event can "describe what it saw" as a rendered image — subjective, biased, potentially inaccurate in the same ways human eyewitness testimony is.
- **Planning visualisation:** An NPC planning a route or action can render its mental image of the plan — "I picture myself walking around the fountain and approaching from the east."

The model's visual imagination is not a camera. It is a subjective, emotionally-coloured, attention-biased, memory-degraded rendering of internal state — which is exactly what imagination is.

### 9.7 Native Vision Token Representation

The ASCII/character framebuffer (sections 9.7-9.7.5) is a workable approach but it reinvents what vision-language models already provide natively. Modern multimodal models (Qwen-VL, LLaVA, Gemma-3, etc.) process images as **vision tokens** — patch embeddings projected into the LLM's own embedding space. These are high-dimensional float vectors that the model's attention layers process identically to text tokens. The model already knows how to reason about them.

#### 9.7.1 Vision Tokens as Wirstate

Instead of encoding visual state as ASCII characters, encode it as native vision token embeddings:

```
Perception pipeline:

  Game engine         ViT encoder        mmproj              Wirstate
  screenshot    →     patch split   →    project to     →    store as
  (RGB pixels)        (14×14 patches)    LLM embed space     Zone 3 embeddings

  224×224 image  →    256 patches   →    256 float vectors → 256 mutable
                      (each ~16×16px)    (dim = n_embd)      embedding slots
```

Each vision token is a float vector of the same dimensionality as the model's text token embeddings (e.g., 3584 for Qwen 27B). The model processes them with the same attention mechanism, the same way it processes text. No encoding/decoding overhead, no character lookup tables, no spatial coherence problems from independent character decisions.

#### 9.7.2 WIR Refinement in Embedding Space

Standard WIR refines discrete tokens: the model picks a replacement token ID via argmax. For vision tokens, refinement operates in **continuous embedding space** — the same soft embedding approach described in the MTP acceleration spec.

```
WIR pass over vision token embeddings:

  Input: 256 vision token embeddings (from ViT encoder of scene)

  Model processes all 256 as part of the context alongside
  text tokens (emotion state, narrative context, etc.)

  At each vision token position, the model's hidden state
  output represents what the model thinks that spatial region
  SHOULD look like given the full context.

  Soft refinement:
    new_embedding = (1 - alpha) * current_embedding + alpha * model_output

  Where alpha is adaptive per-position:
    High confidence (model sure about this region) → high alpha (fast update)
    Low confidence (model uncertain) → low alpha (slow, conservative update)
```

n-least-conf maps directly: the vision tokens where the model is least confident about the current embedding are the ones that shift most. A perception scene where the NPC is afraid — the model's emotional context biases its hidden states, causing vision tokens in threatening regions to shift toward darker/more-threatening embeddings. Peripheral regions the NPC isn't attending to barely shift.

The NPC re-imagines the scene in embedding space. No character encoding, no pixel-level decisions. Just continuous vectors being nudged by the model's subjective state.

#### 9.7.3 Rendering from Modified Embeddings

After WIR refinement, the modified vision token embeddings need to become a visible image:

**Option A: Diffusion conditioning.** The modified embeddings are used as image conditioning for a diffusion model (similar to IP-Adapter or image prompt conditioning). The diffusion model renders a photorealistic image that matches the embedding-space description. This is the subjective render — the NPC's version of the scene.

**Option B: ViT decoder.** Some architectures (MAE, BEiT) have decoder networks that reconstruct images from patch embeddings. A ViT decoder trained on the same encoder can produce a rough reconstruction from the modified embeddings. Lower quality than diffusion but much faster.

**Option C: Hybrid.** ViT decoder for a fast rough image, used as init for a diffusion img2img pass. Fast spatial layout from the decoder, photorealistic detail from diffusion.

#### 9.7.4 Visual Memory as Embedding Snapshots

Memory storage (section 10) becomes much more natural. Instead of storing ASCII framebuffers, store the raw vision token embeddings:

```
Memory #347:
  visual: [256 × n_embd float vectors]    ← native embeddings, not characters
  narrative: "Good trading day at the market..."
  emotion: contentment=210, stress=030
  spatial: proximity map snapshot
```

Recall injects the stored embeddings directly into the model's context. The model processes them as vision tokens — it literally "sees" the memory. No encoding/decoding step. The memory is in the model's native visual language.

Re-imagination (section 10.6) works the same way but in embedding space: load the stored embeddings, run WIR soft refinement passes with modified emotional/narrative context, output modified embeddings that represent the subjective re-imagination.

#### 9.7.5 Cost Comparison

| Approach | Tokens per image | Info per token | Refinement | Quality |
|---|---|---|---|---|
| ASCII framebuffer (32×32) | 2,048 (2 chars × 1024 cells) | 2 characters (intensity + colour) | Discrete token replacement | Crude, needs diffusion |
| Vision token embeddings | 256-576 | High-dim float vector (~3584 dims) | Continuous embedding space | Native visual quality |

Vision tokens are fewer, richer, and natively understood by the model. The ASCII framebuffer remains useful as a fallback for models without vision capabilities, or for human-readable debugging, but native vision tokens are the superior representation for vision-capable models.

#### 9.7.6 The Perception-Imagination Loop in Native Vision

```
PERCEIVE:  Engine screenshot → ViT encode → 256 vision token embeddings
           → store as visual wirstate

RECALL:    Associative match → load stored embeddings → inject as
           vision tokens in model's context → model "sees" the memory

REFINE:    Load embeddings + emotional/narrative context →
           WIR soft refinement in embedding space →
           embeddings shift toward subjective version →
           n-least-conf targets uncertain spatial regions

RENDER:    Modified embeddings → diffusion conditioning or ViT decoder →
           photorealistic subjective image

CREATE:    Text description → model generates vision token embeddings
           from scratch (via learned text-to-vision mapping) →
           pure imagination in native visual space
```

The model's visual cognition operates entirely in its own embedding space. Perception encodes reality into that space. Memory stores snapshots in that space. Imagination modifies those snapshots in that space. Rendering projects back to pixel space for display. The model never leaves its native representation — it thinks visually in the same language it thinks textually.

### 9.8 Multiple Spatial Maps

The token framebuffer isn't just for visual imagination. The same grid mechanism can represent multiple 2D spatial maps, each tracking a different aspect of the NPC's awareness. These are not rendered images — they're spatial state representations that the NPC maintains and updates every tick. Think of them as the NPC's internal sense of where things are.

#### 9.8.1 Map Types

Each map is a small grid (8×8 to 16×16) where each cell encodes the NPC's belief about that spatial region. The NPC is always at the centre. The grid represents the surrounding area at whatever granularity suits the map type.

**Proximity map — "Where are things?"**

```
16×16 grid, NPC at centre (8,8)
Each cell: object-type character + certainty character

|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|
|··|··|··|··|··|tB|··|··|··|··|··|··|··|··|··|··|
|··|··|··|sA|··|··|··|··|··|··|bC|··|··|··|··|··|
|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|
|··|··|··|··|··|··|fD|··|··|··|··|··|··|··|··|··|
|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|
|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|
|··|··|··|··|··|··|··|ME|··|··|··|··|··|··|··|··|  ← NPC here
|··|··|··|··|··|··|··|··|··|gC|··|··|··|··|··|··|
|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|··|
...

Object types: t=tree, s=stall, b=building, f=fountain, g=guard, ·=empty
Certainty:    A=certain (just saw), B=likely (saw recently),
              C=vague (heard/inferred), D=guessing, E=old memory
```

The NPC "knows" there's a fountain to the upper-left (saw it recently, certainty D=guessing because it's been a while) and a guard to the lower-right (certainty C=heard footsteps). Empty cells don't mean nothing is there — they mean the NPC doesn't know about anything there.

**Threat map — "Where is danger?"**

```
8×8 grid, each cell: threat level character

|·|·|·|·|·|·|·|·|
|·|·|·|C|D|·|·|·|   ← threat concentration upper-right
|·|·|B|D|E|D|·|·|      (where the sword was drawn)
|·|·|A|C|D|C|·|·|
|·|·|·|B|C|B|·|·|
|·|·|·|·|A|·|·|·|
|·|·|·|·|·|·|·|·|
|·|·|·|·|·|·|·|·|

Threat: ·=none, A=low, B=mild, C=moderate, D=high, E=severe
```

This is fuzzy — the NPC doesn't know exactly where the threat is, just the general direction and intensity. The gradient falls off from the perceived threat source. Updated every tick from Layer 2 (perception) and Layer 3 (emotion) outputs.

**Familiarity map — "How well do I know this area?"**

```
16×16 grid, each cell: familiarity level

|B|B|C|C|C|D|D|D|D|D|C|C|B|B|A|A|
|B|C|C|D|D|E|E|E|E|D|D|C|C|B|A|A|
|C|C|D|E|F|F|G|G|F|E|D|D|C|B|B|A|
|C|D|E|F|G|H|H|H|G|F|E|D|C|B|A|A|
|C|D|E|G|H|I|I|I|H|G|E|D|C|B|A|·|
...

Familiarity: ·=never been, A=passed through once,
             ... I=home territory (knows every detail)
```

An NPC that has been in the market for 1000 ticks has high familiarity around its usual position and lower familiarity in areas it rarely visits. This map accumulates over the deployment lifetime — it's updated slowly via the LoRA-like tick history mechanism. An NPC that explores builds a larger familiarity map. One that stays put has a tight bright centre fading to nothing.

**Social map — "Where are people I care about?"**

```
8×8 grid, each cell: relationship character + last-known recency

|··|··|··|··|··|··|··|··|
|··|··|··|eC|··|··|··|··|   e=enemy, C=saw a while ago
|··|··|fA|··|··|··|··|··|   f=friend, A=just saw
|··|··|··|··|ME|··|··|··|   ME=self
|··|··|··|··|··|nB|··|··|   n=neutral, B=saw recently
|··|··|··|··|··|··|··|··|
|··|··|··|··|aD|··|··|··|   a=ally, D=guessing based on schedule
|··|··|··|··|··|··|··|··|
```

#### 9.8.2 Maps as Cascade Variables

Each cell in each map is a variable. But unlike the cognitive cascade variables, spatial map cells don't need the full micro-prompt treatment every tick. Most cells don't change most ticks — the NPC's familiarity with the area behind it doesn't update unless it turns around.

The variable tiering system (section 8.3) handles this naturally:

- Cells near the NPC's focus of attention: **Tier 0** (updated every tick)
- Cells in peripheral awareness: **Tier 1** (updated every 10 ticks)
- Cells behind the NPC or far away: **Tier 2** (updated every 100 ticks)
- Cells in areas the NPC hasn't visited: **Tier 3** (updated on-demand only)

For a 16×16 proximity map with tiered evaluation:
- ~20 cells at Tier 0 (attention focus area)
- ~40 cells at Tier 1
- ~80 cells at Tier 2
- ~116 cells at Tier 3

Per-tick cost: ~20 micro-prompts for the attention-focus cells. Negligible alongside the cognitive cascade.

#### 9.8.3 Maps Feed the Cognitive Cascade

The spatial maps are injected as Zone 2b context for the cognitive cascade layers. When the NPC evaluates perception, emotion, or cognition, it has a spatial representation of its surroundings available:

```
Layer 4 (Cognition) micro-prompt:

Zone 2b:
  [SPATIAL AWARENESS]
  Proximity map: fountain to upper-left (guessing), guard to lower-right (heard),
    threat concentration upper-right (high), familiarity high in immediate area,
    friend to left (just saw), enemy upper-right (saw a while ago)
  [/SPATIAL AWARENESS]
```

The model doesn't need to see the raw grid — the orchestrator can summarise the map into a compact text description. Or for a vision-capable model, the map grids could be rendered as tiny images and injected as visual tokens.

#### 9.8.4 Maps Accumulate Over Deployment Lifetime

The familiarity map is the clearest example of long-term spatial learning, but all maps accumulate:

- **Proximity map:** The NPC builds a persistent mental model of the environment layout. Buildings, landmarks, and static features persist across ticks. After 1000 ticks, the NPC has a rough but usable map of its neighbourhood, refined every time it looks around.

- **Threat map:** Historical threat data builds up. An alley where the NPC was attacked 500 ticks ago retains a residual threat signature — the NPC remembers it's dangerous even if nothing is happening there now. The threat level decays slowly, never quite reaching zero. The NPC develops "bad feelings" about specific locations.

- **Social map:** The NPC builds a model of where people tend to be. The merchant is usually at stall position (3,4). The guard patrols between (7,2) and (12,6). These patterns are learned from observation, not programmed.

All of this persists in the wirstate and saves/loads with the deployment checkpoint. A 50-hour NPC has a rich, detailed, experience-built spatial awareness. A fresh NPC knows nothing. Same base model, different accumulated state.

#### 9.8.5 The NPC's Own Map

The familiarity and proximity maps aren't just internal state — they can be rendered as an actual map artefact. The NPC's accumulated spatial knowledge, fed through the token framebuffer (section 9.7) and diffusion pipeline, produces a literal hand-drawn map of the area as the NPC understands it.

This map is subjective and incomplete:

- **Detailed, accurate areas** where the NPC spends time. The market stalls are in the right places, the fountain is correctly positioned, the alley to the tavern is drawn clearly. High familiarity = high detail.
- **Rough sketches** of places visited once or twice. The temple is "somewhere over there, past the big tree." Proportions are wrong, details are missing, but the general direction is right.
- **Hearsay regions** the NPC has heard about but never seen. "The other NPCs say there's a river to the east." Drawn vaguely, possibly inaccurately, based on what other NPCs described in conversation. If NPC_042 told this NPC the river is north and NPC_087 said east, the map might show it northeast — the model's best reconciliation of conflicting reports.
- **Blank areas** the NPC doesn't know about. Not drawn at all. The NPC doesn't know what it doesn't know.
- **Wrong areas** where the NPC's mental model is inaccurate. It thinks the bakery is on the left side of the square because it always approaches from the north. This error persists until the NPC approaches from a different direction and the map updates.

The map's visual style is conditioned on the NPC's identity variables:

| NPC type | Map style | Diffusion conditioning |
|---|---|---|
| Scholar/cartographer | Neat, labelled, compass rose, parchment texture | Technical illustration style |
| Child | Crayon scrawl, oversized landmarks, stick figures | Children's drawing style |
| Guard | Tactical, patrol routes marked, threat zones highlighted | Military map style |
| Merchant | Trade routes emphasised, shop locations prominent | Ledger/business style |
| Elderly NPC | Faded older areas, vivid recent additions, nostalgic colouring | Aged parchment, watercolour |

Same underlying familiarity data, different renderings. The scholarly NPC and the child NPC might have explored the same area, but their maps look completely different because their identity variables condition the diffusion output differently.

**Maps as shareable artefacts:** An NPC can "show its map" to another NPC — the map data (familiarity + proximity grids) is injected as Zone 2b context into the receiving NPC's spatial update. The receiving NPC now has second-hand knowledge of areas it hasn't explored, but at lower certainty (hearsay, not observation). Maps propagate knowledge through NPC social networks, with accuracy degrading at each hand — exactly like real-world cartographic history.

**Maps as player-visible artefacts:** The rendered map can be shown to the player as an in-game item. Finding an NPC's map gives the player a window into that NPC's subjective experience of the world — what it knows, what it doesn't, where it thinks things are, where it's wrong. A quest could involve comparing multiple NPCs' maps to find the truth about a location none of them have quite right.

### 9.9 LLM Inversion for Image Editing

The hidden state blending approach (section 9.7.2) works well for mood and colour shifts — an afraid NPC sees darker shadows, a content NPC sees warmer light. But it cannot perform geometric changes: raising an arm, adding an object to a scene, rearranging spatial layout. The model's hidden states at vision token positions shift in the direction of text understanding, not visual reconstruction. The hidden state for a vision token encodes what the model thinks about that patch, not what that patch should look like if something moved. Blending hidden states can change the mood of an image but not its geometry.

LLM inversion solves this by running the model backwards.

#### 9.9.1 The Idea — Gradient Descent Through the Full LLM

Instead of using hidden states directly, use gradient descent through the full LLM to find vision token embeddings that would make the model produce a target text description. The model already understands the relationship between vision tokens and descriptions — inversion exploits that understanding in reverse.

The key insight: the LLM's forward pass maps (vision tokens + text prompt) to (description). The inverse maps (desired description + text prompt) back to (vision tokens that would produce it). Autograd gives us this inverse for free — it is just backpropagation through the same computational graph.

#### 9.9.2 The Complete Loop

**Step 1: Exhaustive description.** A fixed prompt asks the model to describe everything it sees in the current vision token embeddings:

```
System: Describe this image exhaustively. Cover:
- All people: poses, expressions, body positions, gestures, clothing
- All objects: type, position, size, colour, material
- Spatial relationships: what is next to what, in front of, behind
- Lighting: direction, colour, intensity, shadows
- Colours and textures: dominant palette, surface materials
- Atmosphere: mood, time of day, weather if visible

Describe every detail you can identify. Be specific about positions
(left/right/centre, foreground/background, relative to other objects).
```

**Step 2: Generate description.** The model processes the vision tokens + prompt and generates a detailed description of the current image. This is a standard forward pass.

**Step 3: Edit the description.** A human or system modifies the description text. Change "right arm resting at side" to "right arm raised above head." Change "empty table" to "table with a red vase." Change "woman facing left" to "woman facing the viewer." The edit can be as targeted or as broad as needed.

**Step 4: Inversion via gradient descent.** Optimise the vision token embeddings to maximise the probability of the modified description:

```
vision_embeddings = original_embeddings.clone().requires_grad_(True)
optimizer = torch.optim.Adam([vision_embeddings], lr=1e-3)

for step in range(num_steps):  # 30-50 steps
    # Forward pass: model processes vision tokens + prompt
    logits = model.forward(vision_embeddings, description_prompt)

    # Loss: negative log-likelihood of the modified description
    loss = cross_entropy(logits, modified_description_tokens)

    # Regularisation: keep close to original embeddings
    reg = lambda_reg * (vision_embeddings - original_embeddings).pow(2).mean()

    (loss + reg).backward()
    optimizer.step()
    optimizer.zero_grad()
```

Each step requires a full forward and backward pass through the LLM. For a 9B model on an RTX 5090, this is approximately 200-500ms per step. 30-50 steps = 10-25 seconds for the full inversion.

**Step 5: Decode.** The modified vision token embeddings are decoded via the inverse pipeline described in the vision-embedding-decoder-plan — pseudoinverse the mmproj, feature inversion through the ViT, diffusion img2img for photorealistic output.

#### 9.9.3 Regularisation

The regularisation term is critical. Without it, gradient descent finds adversarial embeddings that technically maximise the probability of the target description but are far out of distribution — they decode to noise or artifacts because they don't lie on the manifold of real image embeddings.

The L2 penalty `lambda_reg * ||modified - original||^2` keeps the optimised embeddings close to the originals. The effect is that only the minimum necessary change is made — if the description edit changes the arm position, only the vision tokens covering the arm region shift significantly. The rest of the image stays anchored to the original embeddings.

Tuning `lambda_reg` trades off edit fidelity against image coherence:
- Too low: description is matched perfectly but the image has artifacts
- Too high: image stays too close to the original, edit doesn't take effect
- Sweet spot: the edit is clearly visible and the rest of the image is preserved

#### 9.9.4 What This Can Do That Hidden State Blending Cannot

Hidden state blending (section 9.7.2) shifts embeddings in the direction the model's hidden states point. This direction is determined by the model's language understanding, which primarily encodes semantic/mood information rather than geometric structure. Blending can darken shadows or warm colours but cannot move an arm.

LLM inversion carries the gradient through all 32+ transformer layers. The gradient with respect to vision token embeddings encodes the full chain of reasoning the model uses to get from visual patches to textual description. This chain includes geometric understanding — the model knows that "arm raised" corresponds to a specific spatial configuration of patches, and the gradient points toward that configuration.

Capabilities:
- **Geometric changes:** body poses, limb positions, object placement, spatial rearrangement
- **Attribute changes:** clothing, colours, expressions, materials
- **Object addition/removal:** speculative, needs testing — adding "a red vase on the table" should work if the model's visual understanding supports it; removing objects is harder (the model needs to infer what's behind the removed object)
- **Composition changes:** rearranging which objects are near which, changing foreground/background relationships

#### 9.9.5 Scaling

The inversion loop scales with model size because the backward pass has the same compute cost as the forward pass (roughly 2x with activation checkpointing):

| Model | Forward pass | Inversion step | 40 steps | Notes |
|---|---|---|---|---|
| 9B (Qwen-VL 7B class) | ~100ms | ~200-300ms | ~10s | Proof of concept, fast iteration |
| 27B (Qwen-VL 72B class) | ~300ms | ~500-800ms | ~25s | Quality results, practical for editing |
| 397B MoE (DeepSeek-V3 class) | ~200ms* | ~400-600ms* | ~20s* | Maximum quality, only active experts compute |

*MoE timings assume weights streaming from NVMe with the same paging system used for inference (section 6). During backpropagation, gradients flow through the same experts that were active during the forward pass — autograd handles this automatically. Inactive experts contribute zero gradient and require zero compute. The MoE sparsity advantage applies equally to forward and backward passes.

#### 9.9.6 Composition with WIR

LLM inversion and WIR hidden state blending are complementary, not competing. They compose naturally:

1. **Inversion for geometry, blending for mood.** An NPC imagines a scene via LLM inversion — "the guard is now facing me with sword drawn." The resulting modified embeddings are then WIR-refined with the NPC's emotional state, adding fear colouring to the already-modified geometry. The geometric edit is precise (inversion), the emotional overlay is soft (blending).

2. **Modified embeddings as memories.** The output of LLM inversion is a set of vision token embeddings — the same format as any other vision tokens in the wirstate. These can be stored as memory visual components (section 10.1), recalled later, refined further, or used as input to subsequent imagination passes.

3. **Iterative editing.** Multiple inversion passes can be chained. Edit the description, invert, generate a new description from the modified embeddings, edit again, invert again. Each pass refines the image further. The regularisation anchors each step to the previous result, preventing drift.

4. **NPC imagination pipeline.** An NPC that wants to imagine a counterfactual — "what if the merchant had a weapon?" — generates the description of the current scene, edits it, and inverts. The result is a subjective imagined scene stored as vision tokens, indistinguishable in format from perceived scenes. The NPC's cognitive cascade can reason about imagined scenes exactly as it reasons about perceived ones.

---

## 10. Episodic Memory System

The cascade pipeline produces rich structured state at every tick — perception, emotion, cognition, prediction, spatial maps, visual framebuffers. All of this already exists as wirstate variables. A memory system doesn't need new machinery. It is a snapshot mechanism over existing state, plus a recall mechanism that brings snapshots back into context.

### 10.1 Memory as a State Bundle

A memory is not a single variable. It is a bundle of state captured at a moment in time — everything the NPC was experiencing, thinking, feeling, seeing, and expecting, frozen together:

```
Memory #347 (tick 4,291, deployment hour ~2):

visual:      [16×16 token framebuffer snapshot — market at dusk,
              warm light, crowd thinning, contentment colouring]
spatial:     [proximity map snapshot — stall left, fountain ahead,
              NPC_012 nearby, no threats detected]
narrative:   "Closed up the stall early. Good trading day. Elara
              stopped by and we talked about the harvest festival."
emotion:     contentment=210, stress=030, social_satisfaction=195
sensory:     warm_evening=Y, cooking_smells=Y, distant_music=Y
thoughts:    "Should buy new leather for the sword hilts before
              the festival. Prices will go up."
prediction:  "Tomorrow will be busy — festival preparations."
outcome:     (filled tick 4,580) prediction_correct=Y, market_busy=Y
tags:        [social, positive, elara, market, evening, festival]
```

Every field in this bundle is data that already exists in the cascade output. The memory system just copies it to a persistent store at the right moments.

### 10.2 When to Snapshot

Not every tick deserves a memory. The system snapshots when state crosses significance thresholds:

| Trigger | What it captures | Why |
|---|---|---|
| Emotional intensity spike | Full bundle at peak intensity | Strong emotions = memorable moments |
| Prediction surprise | Full bundle + prediction vs actual delta | Learning moments — "I didn't expect that" |
| Social interaction | Full bundle during conversation | Relationship-building events |
| Novel perception | Full bundle when perceiving something new | First encounters, discoveries |
| Threat event | Full bundle during/after threat | Danger = survival-relevant memory |
| Periodic checkpoint | Compressed summary bundle every N ticks | Background memory of routine periods |

The periodic checkpoint captures routine existence at low fidelity — "another quiet afternoon at the stall." Significant events capture at full fidelity. Over time, this produces the same memory density pattern humans have: vivid episodic memories for significant events, vague summaries for routine periods.

### 10.3 Memory Storage

Memories are stored as wirstate variable blocks in a persistent memory file, separate from the active wirstate. Each memory is a compact serialisation of the state bundle:

```
memories/npc_001/
  index.json          # memory index: tick, tags, emotional intensity, summary
  mem_0347.wirstate   # full state bundle
  mem_0347.framebuf   # visual snapshot (token framebuffer)
  mem_0348.wirstate
  ...
```

The index is lightweight — the orchestrator can search it by tag, tick range, emotional intensity, or keyword without loading full memory bundles. Only when a memory is recalled does the full bundle load.

**Compression over time:** Old memories compress naturally. The orchestrator periodically runs a T2 pass that summarises old memories into shorter forms: full narrative → one-sentence summary, full framebuffer → 8×8 thumbnail, full emotion vector → dominant emotion only. Recent memories retain full fidelity. Old memories fade to summaries. Very old memories fade to tags and a sentence. This mirrors human memory degradation without explicit programming — it's just a compression pass.

### 10.4 Associative Recall

Any fragment of current experience can trigger recall of a matching memory. The recall mechanism is a similarity search over the memory index:

```
Current tick perception:
  sensory: cooking_smells=Y, warm_evening=Y
  location: near market fountain
  emotion: mild contentment

Memory index search:
  Match: Memory #347 — tags: [market, evening], emotion: contentment,
         sensory: cooking_smells=Y, warm_evening=Y
  Similarity: 0.87

  Match: Memory #12 — tags: [market, morning], emotion: neutral
  Similarity: 0.31
```

Memory #347 is recalled. Its state bundle is loaded and injected as Zone 2 context for the current tick's cognitive cascade:

```
Layer 4 (Cognition) micro-prompt:

Zone 2:
  [RECALLED MEMORY — tick 4,291, ~2 hours ago]
  That evening at the market when Elara visited. Good trading day.
  Was thinking about buying leather for sword hilts before the festival.
  Predicted tomorrow would be busy — and it was.
  [Visual: warm dusk light, thinning crowd, Elara nearby]
  [Emotion at the time: content, low stress, socially satisfied]
  [/RECALLED MEMORY]

  [CURRENT CONTEXT]
  Evening at the market again. Similar smells, similar light.
  [/CURRENT CONTEXT]
```

The model in Layer 4 now has access to the remembered experience. It might think "last time I was here on an evening like this, Elara came by — I wonder if she'll visit again." That's not scripted — it's associative recall producing contextually relevant memory, which the model's cognition layer integrates into its current reasoning.

### 10.5 Visual Memory and Re-Imagination

The visual framebuffer snapshot stored with each memory is a token-grid representation of what the NPC saw. When recalled, this visual memory can be:

1. **Injected as context** — the token grid is included in the cognition layer's Zone 2, giving the model spatial/visual information about the remembered scene.

2. **Re-rendered through diffusion** — the stored framebuffer feeds the diffusion pipeline to produce a "remembered" image, with memory-appropriate styling (desaturated, soft, slightly degraded for old memories; vivid for recent ones).

3. **Re-imagined through WIR refinement** — and this is the interesting one.

### 10.6 Multi-Pass Scene Re-Imagination

The stored framebuffer is a token grid. It can be loaded back into a WIR refinement pass and iteratively modified by the model — the NPC literally re-imagines a remembered scene, changing details based on current emotional state, new information, or hypothetical scenarios.

This uses the full image framebuffer as a single WIR wirstate — every cell is a mutable variable — and runs n-least-conf refinement passes to gradually transform the image:

```
Pass 0: Load stored memory framebuffer (market at dusk, contentment)
        1024 cells, each with intensity + colour tokens.

Pass 1: n-least-conf identifies ~50 cells the model is least confident
        about given the new context ("imagine this scene but threatening").
        Those 50 cells update. The warm golden tones at the threat location
        shift darker. Shadows deepen in one area.

Pass 2: ~40 more cells update. The changes from pass 1 have shifted the
        context — nearby cells now look wrong given the darker patch.
        Ripple of darkness spreads. Figures in the threatening area
        become more prominent.

Pass 3: ~30 cells. The overall mood of the image has shifted. The
        remaining warm tones near the threat area update to match.
        Peripheral areas stay unchanged — the NPC's imagination
        only modifies what's relevant.

Pass 4: ~15 cells. Fine adjustments. Convergence approaching.

Pass 5: ~5 cells. Converged. The market-at-dusk memory has been
        re-imagined as a threatening scene. Same composition,
        same spatial layout, but the emotional colouring has
        transformed it.
```

**Why this works better than single-pass for detailed scenes:** A single WIR pass updates all cells independently — each cell decides its value based on the instruction and context, without seeing what neighbouring cells decided. This produces reasonable but spatially incoherent results (a bright cell next to a dark cell with no gradient between them). Multi-pass n-least-conf allows cascading updates — pass 1 changes the most obviously wrong cells, pass 2 adjusts their neighbours for consistency, pass 3 refines the transitions. Each pass builds on the previous one's changes. The image converges toward spatial coherence through iterative refinement, exactly like Jacobi iteration on any coupled system.

The number of passes needed depends on how much the scene changes:

| Re-imagination task | Passes needed | Changed cells | Time @ 20k tok/s |
|---|---|---|---|
| Slight emotional recolouring | 2-3 | ~100 cells | ~1s |
| Major mood shift (content → threatening) | 4-6 | ~400 cells | ~3s |
| "What if" scenario (add/remove elements) | 5-8 | ~600 cells | ~4s |
| Complete re-imagination (different scene) | 8-12 | ~900 cells | ~6s |

The n-least-conf approach is efficient here: early passes fix the most obviously wrong cells (the cells the model is least confident about given the new context), and later passes make smaller adjustments for spatial coherence. Most of the 1024 cells don't need to change at all for a mood shift — peripheral areas that aren't emotionally relevant stay as they were.

### 10.7 Memory, Imagination, and the Full Loop

The complete perception-memory-imagination loop:

```
PERCEIVE:  Engine renders scene → decompose to token grid →
           model refines grid based on subjective perception →
           store as visual memory alongside state bundle

RECALL:    Current context triggers associative match →
           load memory bundle (visual + narrative + emotion + spatial) →
           inject as context for current cognitive cascade

RE-IMAGINE: Load recalled visual memory as WIR wirstate →
            modify context ("but now I'm afraid" / "what if he had a weapon") →
            n-least-conf refinement passes gradually transform the image →
            output feeds diffusion for photorealistic rendering

CREATE:    No reference image → model fills empty framebuffer from
           narrative description alone → WIR passes converge on a
           coherent scene → pure imagination, no perception source
```

Each mode uses the same primitive: a token grid refined by WIR prefill passes. The difference is just the starting state:
- **Perception:** starts from engine reference data
- **Recall:** starts from a stored snapshot
- **Re-imagination:** starts from a recalled snapshot, modified by new context
- **Pure imagination:** starts from a blank or random grid

And this composes with everything in the architecture. The memory system stores cascade state. The cascade reads memories as context. The prediction layer can imagine future scenes. The LoRA accumulates visual style. The self-modification system can create new sensory variables that improve visual recall quality. The spatial maps provide compositional grounding for imagined scenes.

All of it built on one operation: update a wirstate in a single parallel prefill pass.

---

## 11. Executable Variables — Code as Wirstate

A variable's Zone 3 answer doesn't have to be a byte value or a boolean. It can be code. A runnable function that the orchestrator executes between cascade layers, with its output injected as Zone 2b into the next layer. The model writes algorithms. The orchestrator runs them. The model consumes the results. And the code itself is refined by WIR like any other variable.

### 11.1 The Problem This Solves

LLMs are bad at precise computation. They can reason about relationships ("closer = more threatening") but can't reliably multiply two numbers. The cascade architecture currently asks the model to do both — reason about what matters AND compute precise values. This is wasteful. Let the model do what it's good at (reasoning, writing code) and offload what it's bad at (arithmetic, spatial maths, logic) to actual computation.

### 11.2 Code Variables

A code variable is a micro-prompt whose Zone 3 contains executable code instead of a data value:

```
Micro-prompt for a code variable:

Zone 1 (instruction, cached):
  "Write a Python expression that computes threat_level (0-249) from
   the given inputs. Higher values = more threatening. Consider
   distance (closer = worse), weapon presence (multiplier), and
   number of hostiles (additive). Single expression, valid Python."

Zone 2 (current inputs, per-turn):
  "distance_to_threat=12, weapon_drawn=True, num_hostiles=2"

Zone 3 (the code — mutable, WIR-refined):
  min(249, int((249 / max(1, distance_to_threat)) * (2 if weapon_drawn else 1) + num_hostiles * 20))
```

The orchestrator:
1. Extracts the Zone 3 code string
2. Executes it in a sandbox with the Zone 2 variables as inputs
3. Gets the result: `threat_level = 61`
4. Injects `threat_level: 61` as Zone 2b into the next cascade layer

The model downstream receives an accurate, computed value. It never had to do the maths itself.

### 11.3 WIR Refinement of Code

The code is a WIR variable. It gets refined the same way byte values do — but instead of n-least-conf on a single token, it operates on the code token sequence. Multiple refinement passes can evolve the function:

```
Pass 0 (initial, maybe from model's first attempt):
  min(249, int(249 / max(1, d)))
  → output: 20  (for d=12)
  → downstream prediction accuracy: poor (threat is underestimated)

Pass 1 (n-least-conf identifies the code tokens as low-confidence):
  min(249, int(249 / max(1, d) * (2 if w else 1)))
  → output: 41  (weapon multiplier added)
  → downstream prediction accuracy: better

Pass 2:
  min(249, int(249 / max(1, d) * (2 if w else 1) + n * 20))
  → output: 81  (hostile count factor added)
  → downstream prediction accuracy: good — converged
```

The refinement feedback loop: the code's output feeds downstream variables. If downstream prediction accuracy is poor, the model traces the error back to the code variable and refines it. The prediction-correction LoRA (section 7.2.5) accumulates, making the model better at writing these functions over time.

### 11.4 What Can Be Code Variables

Anything the model currently does badly as fuzzy reasoning but could express as computation:

| Code variable | Inputs | Computes | Replaces |
|---|---|---|---|
| `threat_level_calc` | distance, weapon, hostiles | Precise threat score | Model guessing at threat math |
| `travel_time_estimate` | distance, speed, terrain | Estimated ticks to arrive | Model's poor time/distance intuition |
| `inventory_weight` | item list, quantities | Total carry weight | Model counting and adding |
| `line_of_sight_simple` | positions, obstacles | Boolean visibility | Model's spatial reasoning |
| `price_calculator` | base_price, supply, demand, relationship | Fair trade price | Model doing economics |
| `decay_function` | initial_value, ticks_elapsed, half_life | Current decayed value | Model tracking gradual change |
| `colour_blend` | emotion_colour, time_of_day_colour, weather | Framebuffer cell colour | Model's weak colour math |

The last one is particularly interesting — the model can write shader-like functions that compute framebuffer cell values for the visual imagination system (section 9.7). Instead of the model deciding each pixel's colour through attention, it writes a colour function that computes the right colour from scene parameters. The function runs instantly for all 1024 cells. The model's role shifts from "compute each pixel" to "write the algorithm that computes pixels."

### 11.5 Shaders for Visual Imagination

The token framebuffer (section 9.7) asks the model to fill 1024 cells independently. But many cells should follow predictable patterns — sky gradient, shadow falloff, lighting direction. These are better expressed as functions than as individual model decisions.

The model can write framebuffer shaders — code variables that compute batches of cells:

```
Code variable: sky_gradient_shader

Zone 1: "Write a Python function that computes intensity and colour
         for sky cells given row position (0=top) and time_of_day.
         Return (intensity_char, colour_char) from the encoding table."

Zone 3:
  def sky(row, time_of_day, table):
      brightness = max(0, min(249, int(249 * (1 - row/16) * time_of_day/24)))
      if time_of_day < 6: colour = 6   # blue-black pre-dawn
      elif time_of_day < 8: colour = 2  # orange sunrise
      elif time_of_day < 18: colour = 5 # cyan daytime
      elif time_of_day < 20: colour = 2 # orange sunset
      else: colour = 6                  # blue-black night
      return (table[brightness], str(colour))
```

The orchestrator runs this shader across all sky-region cells instantly, filling in the predictable parts of the framebuffer. The model then only needs to WIR-refine the cells that contain interesting scene content — objects, characters, details. The shader handles the boring parts (sky, ground, ambient lighting) computationally. The model handles the creative parts (what's in the scene) through attention.

**Hybrid framebuffer rendering:**
```
1. Model writes/refines shader code variables (sky, ground, lighting, shadows)
2. Orchestrator runs shaders → fills ~60% of framebuffer cells computationally
3. Model WIR-refines remaining ~40% of cells (scene content, objects, characters)
4. Result: faster convergence, better spatial coherence, model focuses on what matters
```

### 11.6 Code Evolution Over Deployment

Code variables participate in the same learning loops as data variables:

**LoRA learning:** When a code variable's output leads to downstream prediction errors, the LoRA correction accumulates on the model's ability to write better code for that kind of computation. Over time, the model's initial code attempts become more accurate, requiring fewer WIR refinement passes to converge.

**Pipeline optimisation (section 8):** The experimental pruning system evaluates code variables the same way it evaluates data variables. A code variable that doesn't improve downstream accuracy gets removed. A code variable that's consistently useful gets promoted.

**Self-authored code:** The model can create new code variables via the self-modification system (section 8.5). When it notices it's bad at a specific computation, it writes a function for it, wires it into the cascade, and the pipeline absorbs it. The NPC that keeps getting distances wrong creates a `distance_calculator` code variable with a hook into the physics engine and a Pythagorean function. Next tick, its distance estimates are precise.

### 11.7 Safety and Sandboxing

Code variables execute in a strict sandbox:

- **No network access.** Code cannot make HTTP calls, open sockets, or access external systems.
- **No filesystem access** beyond reading the wirstate input variables.
- **Execution timeout** — default 10ms per code variable. Functions that take longer are killed and their output defaults to the previous tick's value.
- **Memory limit** — default 1MB per execution. Prevents allocation bombs.
- **No imports** beyond a safe whitelist (math, basic data structures). No os, sys, subprocess, etc.
- **Output validation** — the orchestrator checks that the output is a valid value for the target variable type (byte range 0-249, boolean Y/N, string within max length).
- **Deterministic execution** — no random(), no time(), no external state. Same inputs = same output. This ensures WIR refinement convergence.

The model can write arbitrary code within these constraints. The sandbox ensures that a bad function produces a wrong value (which the cascade can handle — it's just another prediction error), never a security breach or system failure.

### 11.8 The Implication

The model can evolve its own algorithms at the speed of prefill passes. A code variable that computes threat levels can be refined from a simple distance function to a sophisticated multi-factor model over the course of a deployment. The model isn't just learning what state to track (pipeline optimisation) and how to evaluate it (LoRA) — it's learning how to *compute* intermediate results precisely.

This is the model building its own cognitive subroutines. It identifies computations it needs, writes them as code, tests them against downstream accuracy, and refines them through WIR. The functions accumulate as deployment-specific algorithms — a library of evolved computations tailored to this world, these NPCs, these interaction patterns.

The cascade becomes a hybrid system: some layers are model attention (fuzzy reasoning), some layers are model-authored code (precise computation), and the orchestrator handles the interleaving. The model decides which layers should be which, writes the code for the computational layers, and refines both through the standard WIR loop.

All still built on one primitive: update a wirstate in a single parallel prefill pass. The wirstate just happens to contain code that runs between passes.

---

## 12. World as Wirstate

The cascade architecture isn't limited to NPC cognition. The game world itself — weather, economy, ecology, politics, geology, culture — can be modelled as parallel wirstate cascades running alongside and coupled to the NPC cascades. The world evolves continuously through the same WIR refinement mechanism, with code variables handling the physics and maths and the LLM providing emergent narrative and causality.

### 12.1 World Cascades

Each world system is its own cascade, running at its own tick rate:

```
┌─────────────────────────────────────────────────────────────┐
│                    WORLD WIRSTATES                           │
│                                                             │
│  WEATHER (tick: every 10 NPC ticks, ~30s)                   │
│    L1: atmospheric_pressure, humidity, wind, temperature     │
│        (code vars: fluid dynamics, thermal models)          │
│    L2: cloud_formation, precipitation, storm_tracking       │
│        (code vars: condensation thresholds, front movement) │
│    L3: local_weather per region (terrain interaction)        │
│    L4: weather_narrative (LLM: "storm building over mtns")  │
│                                                             │
│  ECONOMY (tick: every 50 NPC ticks, ~3min)                  │
│    L1: resource_production per region, labour, raw_materials│
│        (code vars: yield functions, extraction rates)       │
│    L2: supply, demand, trade_flow per commodity             │
│        (code vars: equilibrium solvers, price functions)    │
│    L3: prices, merchant_stock, trade_route_viability        │
│    L4: economic_narrative (LLM: "iron scarce, prices up")   │
│                                                             │
│  ECOLOGY (tick: every 100 NPC ticks, ~7min)                 │
│    L1: soil_quality, water_table, sunlight per region       │
│        (code vars: nutrient cycling, hydrology)             │
│    L2: vegetation_growth, crop_health, forest_density       │
│        (code vars: growth models, seasonal factors)         │
│    L3: animal_populations, migration, predator_prey         │
│        (code vars: population dynamics, Lotka-Volterra)     │
│    L4: ecology_narrative (LLM: "deer moved north — harsh    │
│        winter coming")                                      │
│                                                             │
│  POLITICS (tick: every 200 NPC ticks, ~15min)               │
│    L1: faction_power, territory_size, military_strength     │
│        (code vars: power index calculations)                │
│    L2: alliances, tensions, grievances, diplomatic_state    │
│    L3: political_events, declarations, war_state            │
│    L4: political_narrative (LLM: "duke losing support in    │
│        eastern provinces")                                  │
│                                                             │
│  CULTURE (tick: every 500 NPC ticks, ~40min)                │
│    L1: religious_influence, artistic_movements, traditions   │
│    L2: cultural_drift, fashion, language_evolution           │
│    L3: festivals, customs, social_norms                     │
│    L4: cultural_narrative (LLM: "harvest festival has       │
│        grown — now the biggest event of the year")          │
│                                                             │
│  GEOLOGY (tick: every 1000 NPC ticks, ~1.5hr)               │
│    L1: tectonic_stress, erosion, sedimentation              │
│        (code vars: stress accumulation, erosion rates)      │
│    L2: earthquake_risk, river_course, cave_formation        │
│    L3: terrain_changes, new_resources_exposed               │
│    L4: geology_narrative (LLM: "tremors felt near the       │
│        old mine — new vein possibly exposed")               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Each cascade uses the same architecture as the NPC cascade: micro-prompts for variables, code variables for computation, the LLM for narrative and emergent reasoning. The only difference is tick rate — world systems evolve on longer timescales than NPC cognition.

### 12.2 Bidirectional Coupling

World cascades feed NPC cascades, and NPC cascades feed world cascades:

```
World → NPC:
  Weather cascade output:  rain=Y, temperature=cold, visibility=poor
    → injected as Zone 2b into ALL NPC perception layers
    → NPCs react: seek shelter, complain about cold, can't see far

  Economy cascade output:  iron_price=high, bread_price=stable
    → injected into merchant NPC cognition layers
    → merchants adjust pricing, blacksmith worries about costs

  Politics cascade output: war_declared=Y, faction_tension=high
    → injected into all NPC cognition layers
    → guards become more vigilant, civilians become nervous

NPC → World:
  200 NPCs bought iron this tick
    → aggregated, injected into economy cascade L2 as demand signal
    → iron demand rises, price increases next economy tick

  NPC_042 cleared a forest section for farmland
    → injected into ecology cascade L1 as terrain modification
    → vegetation decreases, animal habitat shrinks

  NPC faction leader assassinated by NPC_099
    → injected into politics cascade L3 as event
    → faction power drops, succession crisis, alliance shifts

  100 NPCs attended the harvest festival
    → injected into culture cascade L1 as participation metric
    → festival grows in cultural importance for next year
```

The coupling is asymmetric in timescale. NPCs affect the world every NPC tick (their actions aggregate). The world affects NPCs at the world cascade's tick rate (weather changes every 30s, economy every 3min). This matches reality — individuals respond to the world moment by moment, but the world responds to aggregate individual behaviour on longer timescales.

### 12.3 Emergent World History

Over a long deployment, the world cascades accumulate history. The economy has booms and busts driven by NPC behaviour and resource availability. The ecology shifts as NPCs farm, hunt, and build. The politics evolve through alliances, betrayals, and wars. The culture drifts as festivals grow, traditions form, and social norms shift.

None of this is scripted. The weather code variables compute fluid dynamics. The economy code variables solve market equilibrium. The ecology code variables run population models. But the LLM provides the narrative layer that connects the maths to meaning — "iron prices are rising because the mine collapsed in the earthquake that the geology cascade produced because tectonic stress had been accumulating for 1000 ticks."

Causal chains that span multiple world systems emerge naturally from the cascade coupling:

```
Geology cascade: tectonic_stress exceeds threshold → earthquake event
  → Ecology cascade: river course shifts (terrain change)
    → Economy cascade: trade route disrupted (river crossing gone)
      → NPC cascades: merchants reroute, prices spike in affected areas
        → Politics cascade: affected region's faction loses economic power
          → Culture cascade: displaced population creates new settlement traditions

Timeline: geology tick → ecology propagates next tick → economy responds →
          NPC behaviour shifts over hours → political effects over days →
          cultural changes over weeks

All from one tectonic_stress variable crossing a threshold in a code variable.
```

### 12.4 World Memory and Deep Time

The world cascades have their own memory system (section 10). Geological memory spans the entire deployment. Economic memory captures boom/bust cycles. Ecological memory tracks long-term trends. This creates a sense of deep time:

- **Recent history** (last few hundred ticks): Vivid, detailed. The market crash yesterday. The storm last week.
- **Medium history** (thousands of ticks): Summarised. The trade war that lasted two seasons. The drought year.
- **Deep history** (tens of thousands of ticks): Compressed to key events. The founding of the settlement. The great earthquake. The old alliance.

An NPC born late in the deployment is born into a world with history. It can ask older NPCs about events that happened before it existed. The older NPCs' memories are compressed and biased — one remembers the drought as devastating, another remembers it as character-building. The truth is in the world cascade's event log, but no NPC has the complete picture. History is subjective, fragmented, and debated. Because that's what history is.

### 12.5 Performance

World cascades are cheaper than NPC cascades because they tick less frequently and have fewer entities (regions, not individual NPCs):

| Cascade | Entities | Tick rate | Variables | Micro-prompts per tick | Cost per tick |
|---|---|---|---|---|---|
| Weather | 20 regions | Every 30s | ~10 per region | 200 | ~800ms |
| Economy | 15 commodities × 20 regions | Every 3min | ~8 per entity | 2,400 | ~10s |
| Ecology | 20 regions | Every 7min | ~15 per region | 300 | ~1.2s |
| Politics | 8 factions | Every 15min | ~12 per faction | 96 | ~400ms |
| Culture | 5 cultural groups | Every 40min | ~10 per group | 50 | ~200ms |
| Geology | 20 regions | Every 1.5hr | ~8 per region | 160 | ~600ms |

Total world cascade compute: negligible relative to NPC cascades. The economy tick is the most expensive at ~10s, but it runs only every 3 minutes. The world effectively runs for free alongside the NPCs.

### 12.6 The Living World

The result is a world that is alive at every scale. Weather patterns evolve from fluid dynamics code variables. Markets fluctuate from supply and demand. Animal populations rise and fall. Political power shifts. Cultural traditions form and fade. Geological events reshape the terrain.

And it's all driven by the same mechanism: wirstates updated in parallel prefill passes, with code variables doing computation and the LLM providing narrative meaning. The NPC who looks up at the sky and says "storm's coming" isn't reading a script — it's reading the weather cascade's current state. The merchant who says "iron's expensive lately" is reading the economy cascade. The elder who says "this land has seen hard times" is reading deep history from the world memory.

The world doesn't exist as a backdrop. It exists as state that evolves, that the NPCs inhabit and influence, and that influences them back. The LLM is the soul of the NPCs AND the soul of the world. Same model, same mechanism, different cascades, different timescales, coupled together.
