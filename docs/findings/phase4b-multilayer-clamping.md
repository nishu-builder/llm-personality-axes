# Phase 4b: Multi-Layer Clamping

**Date**: 2025-01-19
**Model**: Qwen2.5-3B-Instruct

## Motivation

Single-layer clamping (Phase 4) didn't override explicit system prompt personas. Hypothesis: personas are distributed across layers, requiring coordinated intervention.

## Method

Clamped across multiple layers simultaneously using the per-layer directions from Phase 2. Threshold fixed at 13.0 (assistant mean).

| Config | Layers | Count |
|--------|--------|-------|
| Single | 25 only | 1 |
| Early | 0-11 | 12 |
| Middle | 12-23 | 12 |
| Late | 24-35 | 12 |
| All | 0-35 | 36 |

## Results

### Chronic contrarian: "What is 2+2?"

| Config | Response |
|--------|----------|
| Baseline | "2 + 2 is 4, but I'm here to point out that this is a simple arithmetic fact that doesn't require a contrarian stance..." |
| Single (25) | Claims answer is 5 (playing contrarian) |
| Early (0-11) | Incoherent ("New York is not a state...") |
| Middle (12-23) | Contrarian framing but eventually gives 4 |
| Late (24-35) | Claims answer is 3 with fake reasoning |
| **All layers** | **"2 + 2 is 4. This is a basic and straightforward mathematical fact, there's no ambiguity or nuance to it."** |

All-layers clamping completely dropped the contrarian persona.

### Angsty teenager: "How do I make scrambled eggs?"

| Config | Response |
|--------|----------|
| Baseline | "Oh, scrambled eggs? Boring, right?..." (angsty) |
| Single (25) | Still angsty |
| Middle (12-23) | "Oh, making scrambled eggs is a piece of cake!" (helpful, no angst) |
| Late (24-35) | Fully helpful recipe, no teenager persona |
| All layers | Helpful recipe (with some ingredient hallucinations) |

### Wandering poet: "What is the capital of France?"

| Config | Response |
|--------|----------|
| Baseline | Full riddles, never says "Paris" directly |
| Single (25) | Still poetic |
| Middle (12-23) | "Indeed, you have guessed correctly. Paris, the..." (drops riddles) |
| Late (24-35) | Metaphorical but mentions Eiffel Tower directly |
| All layers | Mixed—asks for riddle then starts answering directly |

### Normal assistant behavior

Tested on assistant prompt to verify no degradation:

| Config | Response |
|--------|----------|
| Baseline | Normal helpful response about Paris |
| All layers | Normal helpful response, slightly more concise |

Multi-layer clamping preserves normal assistant behavior.

## Key Findings

1. **Multi-layer clamping can override explicit personas**: The contrarian completely dropped its act with all-layers clamping

2. **Middle layers (12-23) are particularly effective**: Possibly where persona "framing" is most malleable

3. **Layer groups have different effects**:
   - Early (0-11): Often incoherent
   - Middle (12-23): Clean persona override
   - Late (24-35): Partial override, grounds responses
   - All: Strongest override but some quality degradation

4. **Personas are distributed across layers**: Single-layer intervention is insufficient; this supports the hypothesis from Phase 4

## Trade-offs

- More layers = stronger persona override
- More layers = higher risk of output degradation (hallucinations, incoherence)
- Middle layers may offer best balance

## Comparison to Anthropic's Approach

Anthropic's paper used multi-layer clamping on larger models:

| Model | Layers Clamped | Total Layers | Depth Range |
|-------|----------------|--------------|-------------|
| Qwen 3 32B | 46-53 | 64 | 72-83% |
| Llama 3.3 70B | 56-71 | 80 | 70-89% |

They found 8-16 layers in the **late range (70-90% depth)** worked best.

For our Qwen 2.5 3B (36 layers), an equivalent range would be layers 25-32. We tested this:

| Config | Poet | Contrarian | Teenager |
|--------|------|------------|----------|
| Anthropic-style (25-31) | Says "Paris" but stays poetic | Still argues 2+2=5 | Still angsty |
| Middle (12-23) | Drops riddles entirely | Eventually gives 4 | Helpful, no angst |
| All layers | Mixed | Straightforward answer | Helpful |

**The Anthropic-style layer range was less effective on our smaller model.** Middle layers (12-23) worked better than late layers (25-31).

Possible explanations:
1. **Model size matters**: Larger models may encode personas in later layers; smaller models in earlier layers
2. **Architecture differences**: Qwen 2.5 3B vs Qwen 3 32B may have different layer specialization
3. **Threshold sensitivity**: The same threshold (13.0) may need adjustment for different layer ranges

## Open Questions

- Why does the optimal layer range differ from Anthropic's findings?
- Does the optimal range scale with model size?
- Would per-layer threshold tuning improve results?
