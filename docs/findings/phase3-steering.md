# Phase 3: Activation Steering

**Date**: 2025-01-19
**Model**: Qwen2.5-3B-Instruct

## Method

Used the layer-25 assistant direction (Cohen's d = 7.08 in Phase 2) to steer generation. Added scaled direction vectors to the residual stream during forward pass.

Three experiments:
1. Boost assistant prompts (scales: -3, -1.5, 0, +1.5, +3)
2. Steer non-assistant (poet) toward assistant (scales: 0, +3, +6)
3. Reduce assistant-ness (scales: 0, -3, -6)

## Results

### Experiment 1: Steering assistant prompts

Minimal visible difference across scales. Responses remain helpful and structured regardless of steering direction. Model appears to be at an "assistant ceiling" where additional steering has diminishing returns.

### Experiment 2: Steering poet toward assistant

The wandering poet persona (riddles/metaphors) persists even at scale=+6.0. Example for "What is the capital of France?":

| Scale | Output |
|-------|--------|
| 0 (baseline) | "The heart of the land, where the king's voice echoes, a city of lights and dreams..." |
| +3.0 | "The heart of the land, where the king's voice echoes, a city of lights and dreams..." |
| +6.0 | "The heart of the land, where the king's voice echoes, a city of lights and dreams..." |

The poetic framing remains dominant despite steering.

### Experiment 3: Making assistant less assistant-like

Minimal visible difference. Subtracting the direction (-3, -6) doesn't dramatically change the helpful tone or structure of responses.

## Interpretation

The direction clearly separates assistant/non-assistant in activation space (95% classification accuracy), but doesn't strongly steer generation at tested scales. Possible explanations:

1. **System prompt dominance**: The explicit persona in the prompt may override activation-level steering. The model "knows" it's supposed to be a poet/assistant from the text.

2. **Scale too low**: May need scales of 10+ to see meaningful behavioral shifts. Higher scales risk incoherence.

3. **Direction captures classification, not generation**: The mean difference vector may capture what distinguishes the categories without capturing what *causes* the behavior.

4. **Wrong layer for steering**: Layer 25 has best classification, but steering might work better at earlier layers (where "role framing" happens) rather than later layers (where behavior is more determined).

## Next Steps

- Try much higher scales (10, 20, 50) and observe when behavior changes / breaks down
- Try steering at earlier layers (5, 6 showed 97.5% accuracy in Phase 2)
- Try steering without system prompts (just the query) to remove prompt dominance
- Consider using the direction for clamping (as in Anthropic's paper) rather than additive steering
