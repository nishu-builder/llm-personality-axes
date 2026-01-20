# Phase 4: Capping (Anthropic Approach)

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

Capping did not override explicit persona instructions at the thresholds we tested. Even aggressive capping (threshold=13.3) left:
- Wandering poet still speaking in riddles
- Contrarian still disagreeing
- Conspiracy theorist still connecting dots

**Hypotheses for why capping didn't work here:**

1. **Single-layer intervention is insufficient**: We only capped at layer 25. The persona may be encoded across multiple layers, requiring coordinated intervention.

2. **Threshold not aggressive enough**: Our most aggressive threshold (13.3) pushed projections to the assistant mean. Higher thresholds might produce visible effects (though risk incoherence).

3. **Direction captures correlation, not causation**: The mean difference vector distinguishes the categories but may not represent the causal mechanism that produces the behavior.

4. **Persona is high-dimensional**: The model may implement personas using many directions, not just projection onto our single axis. Capping one direction leaves others free.

5. **Residual stream vs attention patterns**: Capping residual stream activations may not affect attention patterns that were already computed based on the system prompt.

We haven't ruled out that stronger or multi-layer capping could override explicit prompts.

### Bare prompts (no system instruction)

Minimal visible differences. The instruct-tuned model is already assistant-like by default—it's already projecting positively on the assistant direction.

### Normal assistant prompts

Capping preserves behavior perfectly. Outputs are essentially identical across all threshold levels. This is the intended behavior: minimal intervention when model is behaving normally.

## Conclusions

1. **Capping is safe**: At tested thresholds, it doesn't degrade normal assistant behavior
2. **Single-layer capping didn't override explicit prompts**: At tested thresholds, system prompt personas persisted
3. **Hypothesis #1 (multi-layer intervention needed) worth testing**: See [Phase 4b](phase4b-multilayer-capping.md)

## Next

[Phase 4b: Multi-Layer Capping](phase4b-multilayer-capping.md) tests whether coordinated intervention across multiple layers can override explicit personas. Spoiler: it works.
