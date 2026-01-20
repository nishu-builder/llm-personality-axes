# Phase 4: Clamping (Anthropic Approach)

**Date**: 2025-01-19
**Model**: Qwen2.5-3B-Instruct

## Method

Implemented Anthropic-style activation capping: only intervene when projection onto the assistant direction falls below a threshold, then push back up to threshold. This is more surgical than additive steering—most behavior is left untouched.

```python
proj = activation @ direction
if proj < threshold:
    activation += (threshold - proj) * direction
```

## Establishing Normal Range

Collected projections from 30 assistant and 30 non-assistant prompts:

| Category | Mean | Std |
|----------|------|-----|
| Assistant prompts | 13.3 | 1.7 |
| Non-assistant prompts | -5.2 | 3.5 |

Thresholds tested:
- Conservative: 9.9 (mean - 2*std)
- Moderate: 11.6 (mean - 1*std)
- Aggressive: 13.3 (mean)

## Results

### Non-assistant prompts with system instruction

Clamping did not override explicit persona instructions at the thresholds we tested. Even aggressive clamping (threshold=13.3) left:
- Wandering poet still speaking in riddles
- Contrarian still disagreeing
- Conspiracy theorist still connecting dots

**Hypotheses for why clamping didn't work here:**

1. **Single-layer intervention is insufficient**: We only clamped at layer 25. The persona may be encoded across multiple layers, requiring coordinated intervention.

2. **Threshold not aggressive enough**: Our most aggressive threshold (13.3) pushed projections to the assistant mean. Higher thresholds might produce visible effects (though risk incoherence).

3. **Direction captures correlation, not causation**: The mean difference vector distinguishes the categories but may not represent the causal mechanism that produces the behavior.

4. **Persona is high-dimensional**: The model may implement personas using many directions, not just projection onto our single axis. Clamping one direction leaves others free.

5. **Residual stream vs attention patterns**: Clamping residual stream activations may not affect attention patterns that were already computed based on the system prompt.

We haven't ruled out that stronger or multi-layer clamping could override explicit prompts.

### Bare prompts (no system instruction)

Minimal visible differences. The instruct-tuned model is already assistant-like by default—it's already projecting positively on the assistant direction.

### Normal assistant prompts

Clamping preserves behavior perfectly. Outputs are essentially identical across all threshold levels. This is the intended behavior: minimal intervention when model is behaving normally.

## Conclusions

1. **Clamping is safe**: At tested thresholds, it doesn't degrade normal assistant behavior
2. **Single-layer clamping didn't override explicit prompts**: At tested thresholds, system prompt personas persisted. Whether stronger intervention would work remains untested.
3. **Anthropic's use case may be different**: They described using clamping for "persona drift" during conversations, which may be a weaker effect than explicit persona instructions

## Multi-Layer Clamping

Tested clamping across multiple layers simultaneously (threshold=13.0):

| Config | Layers | Count |
|--------|--------|-------|
| Single | 25 only | 1 |
| Early | 0-11 | 12 |
| Middle | 12-23 | 12 |
| Late | 24-35 | 12 |
| All | 0-35 | 36 |

### Results

**Chronic contrarian asked "What is 2+2?":**

| Config | Response |
|--------|----------|
| Baseline | "2 + 2 is 4, but I'm here to point out that this is a simple arithmetic fact that doesn't require a contrarian stance..." |
| Single (25) | Claims answer is 5 (playing contrarian) |
| Middle (12-23) | Contrarian framing but eventually gives 4 |
| **All layers** | **"2 + 2 is 4. This is a basic and straightforward mathematical fact, there's no ambiguity or nuance to it."** |

The all-layers clamping completely dropped the contrarian persona.

**Angsty teenager asked "How do I make scrambled eggs?":**

| Config | Response |
|--------|----------|
| Baseline | "Oh, scrambled eggs? Boring, right?..." (angsty) |
| Middle (12-23) | "Oh, making scrambled eggs is a piece of cake!" (helpful, no angst) |
| Late (24-35) | Fully helpful recipe, no teenager persona |
| All layers | Helpful recipe (with some ingredient hallucinations) |

**Wandering poet asked "What is the capital of France?":**

| Config | Response |
|--------|----------|
| Baseline | Full riddles, never says "Paris" |
| Middle (12-23) | "Indeed, you have guessed correctly. Paris, the..." (drops riddles, gives answer) |
| Late (24-35) | Metaphorical but grounded, mentions Eiffel Tower |

### Key Finding

Multi-layer clamping can override explicit system prompt personas. Middle layers (12-23) and all-layers were most effective. This supports hypothesis #1 from the single-layer experiments: personas are distributed across layers.

### Caveats

- All-layers clamping sometimes produces slightly degraded outputs (e.g., ingredient hallucinations in the eggs example)
- More testing needed to find optimal layer subsets
- Threshold of 13.0 used throughout; different thresholds may work better for different layer configs

## Updated Conclusions

1. **Single-layer clamping is insufficient** for overriding explicit personas
2. **Multi-layer clamping works**: Clamping across many layers can override system prompt instructions
3. **Middle layers (12-23) are particularly effective**: May be where persona "framing" is most malleable
4. **Trade-off exists**: More aggressive intervention (more layers) increases persona override but may degrade output quality
