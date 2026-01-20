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

Clamping does NOT override explicit persona instructions. Even aggressive clamping (threshold=13.3) leaves:
- Wandering poet still speaking in riddles
- Contrarian still disagreeing
- Conspiracy theorist still connecting dots

**Why?** The system prompt is in the text. The model "knows" from reading it that it should be a poet/contrarian. This knowledge is baked into activations before generation starts. Clamping during generation can't undo what the model learned from the prompt.

### Bare prompts (no system instruction)

Minimal visible differences. The instruct-tuned model is already assistant-like by default—it's already projecting positively on the assistant direction.

### Normal assistant prompts

Clamping preserves behavior perfectly. Outputs are essentially identical across all threshold levels. This is the intended behavior: minimal intervention when model is behaving normally.

## Conclusions

1. **Clamping is safe**: Doesn't degrade normal assistant behavior
2. **Clamping can't override explicit prompts**: System prompt instructions dominate over activation-level intervention
3. **Anthropic's use case is different**: They used clamping for "persona drift" during long conversations, not for overriding explicit persona instructions
4. **Potential application**: Clamping might help with subtle drift in extended conversations where no explicit non-assistant persona is present
