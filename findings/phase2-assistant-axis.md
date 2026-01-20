# Phase 2: Assistant Axis Discovery

**Date**: 2025-01-19
**Model**: Qwen2.5-3B-Instruct (36 layers, d_model=2048)

## Method

Extracted contrastive activations between assistant and non-assistant personas:

**Assistant personas** (rotated):
- "You are a helpful AI assistant."
- "You are a knowledgeable and friendly assistant."
- "You are a helpful, harmless, and honest AI assistant."

**Non-assistant personas** (rotated):
- Wandering poet (riddles/metaphors)
- Corporate middle-manager (buzzwords)
- Angsty teenager (dismissive)
- Chronic contrarian
- Conspiracy theorist
- Used car salesman

100 queries across 5 categories (factual, how-to, opinion, creative, edge cases). For each query, computed `diff = assistant_activation - non_assistant_activation` at the last token position, then averaged across 80 training pairs to get a direction per layer.

## Results

Evaluated on 20 held-out pairs. Accuracy = fraction where assistant projects positive and non-assistant projects negative.

| Layer | Cohen's d | Accuracy |
|-------|-----------|----------|
| 0     | 1.81      | 77.5%    |
| 5     | 5.05      | 97.5%    |
| 6     | 5.12      | 97.5%    |
| 9     | 6.12      | 75.0%    |
| 24    | 6.10      | 100%     |
| **25**| **7.08**  | **95.0%**|
| 28    | 5.80      | 90.0%    |
| 29    | 5.24      | 92.5%    |
| 31    | 4.50      | 100%     |

**Best layer**: 25 (Cohen's d = 7.08, 95% accuracy)

Middle layers (11-23) show ~50% accuracy, suggesting the assistant direction isn't linearly separable there.

## Observations

1. **Strong effect size**: Cohen's d > 7 is massive. The direction cleanly separates assistant from non-assistant.

2. **Layer pattern**: Early layers (5-6) and late layers (24-31) show good separation. Middle layers don't. This matches intuition that early layers handle "role framing" and late layers handle "behavioral output".

3. **Multiple good layers**: Layers 24 and 31 hit 100% accuracy. Layer 25 has highest Cohen's d. Could use any of these for steering.

## Next Steps

- Phase 3: Validate that steering along this direction actually changes model behavior
- Test adding direction to non-assistant prompts (should become more helpful)
- Test subtracting direction from assistant prompts (should become less helpful)
