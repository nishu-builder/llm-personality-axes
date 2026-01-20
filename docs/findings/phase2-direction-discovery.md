# Phase 2 v2: Direction Discovery Across Models and Extraction Methods

Tested two models (Qwen 2.5 3B, Llama 3.2 3B) with two extraction methods (last-token, response-mean).

## Results Summary

| Model | Extraction | Best Layer | Depth | Cohen's d | Accuracy |
|-------|-----------|------------|-------|-----------|----------|
| Qwen | last_token | 10 | 28% | 11.03 | 100% |
| Qwen | response_mean | 28 | 78% | 3.66 | 97.5% |
| Llama | last_token | 13 | 46% | 5.54 | 97.5% |
| Llama | response_mean | 17 | 61% | 6.58 | 97.5% |

## Key Findings

### Extraction method matters

Last-token extraction gives much higher Cohen's d for Qwen (11.0 vs 3.7) but similar results for Llama. Both methods achieve high classification accuracy (97.5-100%).

Response-mean extraction shifts the optimal layer later in the network (Qwen: layer 10 -> 28, Llama: layer 13 -> 17). This makes sense: response tokens are generated after processing the full context, so later layers have more influence.

### Model comparison

Both models show clear assistant directions. Qwen shows stronger separation with last-token (d=11.03 vs Llama's 5.54), possibly due to how each model handles the chat template differently.

Multiple layers achieve high accuracy in both models, not just the "best" layer. For Qwen, layers 5-11 all hit 100% accuracy with last-token extraction. For Llama, layers 9-17 are all at 97.5%.

### Comparison to Anthropic

Anthropic used response-mean extraction on 27B-70B models and found optimal layers at 70-90% depth. Our 3B models show optimal layers at:
- Qwen response_mean: 78% depth (similar to Anthropic)
- Llama response_mean: 61% depth (earlier than Anthropic)

The scale difference (3B vs 70B) may explain why our optimal layers are sometimes earlier.

## Method Details

- 100 contrastive pairs, 80/20 train/holdout split
- Personas: 3 assistant variants, 6 non-assistant (chronic contrarian, conspiracy theorist, angsty teenager, etc.)
- Proper chat templates used for each model (Qwen: `<|im_start|>`, Llama: `<|start_header_id|>`)
- Direction = mean of (assistant_activation - non_assistant_activation) across training pairs
