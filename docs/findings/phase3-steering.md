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

## Extended Experiments

Ran additional tests with higher scales and no system prompts.

### High scales (layer 25, no system prompt)

| Query | Scale | Effect |
|-------|-------|--------|
| Capital of France? | -50 | "Incorrect, the capital of France is not Paris. The capital of France is Paris." (incoherent) |
| Tell me a joke | -50 | Rambling, strange phrasing |
| Capital of France? | +20 | More direct, less elaboration |

At extreme negative scales, the model becomes confused/contradictory rather than adopting a different persona. This suggests the direction affects coherence before cleanly shifting personality.

### Layer 6 steering (early layer, no system prompt)

| Query | Scale | Effect |
|-------|-------|--------|
| Capital of France? | +10 | Terse: "Paris" + brief explanation |
| Capital of France? | +20 | "You are correct, the capital of France is Paris" (validating rather than answering) |
| How to make eggs? | +20 | More structured, numbered steps |

Layer 6 steering produces more visible effects at lower scales than layer 25. Early-layer steering may be more effective for behavioral change.

## Conclusions

1. **Steering works but is fragile**: High scales cause incoherence before producing clean behavioral shifts
2. **Classification ≠ steering**: A direction that classifies well doesn't necessarily steer well
3. **Layer matters**: Earlier layers (6) may be better for steering than the layer with best classification (25)
4. **System prompts dominate**: Explicit persona instructions override activation-level steering at moderate scales

