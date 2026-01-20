# Direction Discovery

Tested two models (Qwen 2.5 3B, Llama 3.2 3B) with two extraction methods (last-token, response-mean).

## Results

| Model | Extraction | Best Layer | Depth | Cohen's d | Accuracy |
|-------|-----------|------------|-------|-----------|----------|
| Qwen | last_token | 10 | 28% | 11.03 | 100% |
| Qwen | response_mean | 28 | 78% | 3.66 | 97.5% |
| Llama | last_token | 13 | 46% | 5.54 | 97.5% |
| Llama | response_mean | 17 | 61% | 6.58 | 97.5% |

## Observations

### Extraction method

Last-token extraction gives higher Cohen's d for Qwen (11.0 vs 3.7) but similar results for Llama. Both methods achieve high classification accuracy on our holdout set (97.5-100%).

Response-mean extraction shifts the optimal layer later in the network (Qwen: layer 10 -> 28, Llama: layer 13 -> 17). One possible explanation: response tokens are generated after processing the full context, so later layers may have more influence on response-specific features.

### Model differences

Qwen shows stronger separation with last-token (d=11.03 vs Llama's 5.54). We don't have a clear explanation for this difference - it could relate to architecture, training data, or how each model represents the chat template.

Multiple layers achieve high accuracy in both models, not just the "best" layer. For Qwen, layers 5-11 all hit 100% accuracy with last-token extraction. For Llama, layers 9-17 are all at 97.5%.

### Comparison to Anthropic

Anthropic used response-mean extraction on 27B-70B models and found optimal layers at 70-90% depth. Our results:
- Qwen response_mean: 78% depth (similar)
- Llama response_mean: 61% depth (earlier)

The difference could be due to model scale (3B vs 70B), architecture differences, or our smaller persona set (6 vs 275).

## Caveats

- Small holdout set (20 samples) means accuracy numbers have high variance
- We tested 6 non-assistant personas vs Anthropic's 275 - our direction may be less general
- High Cohen's d on classification doesn't guarantee the direction will be useful for steering

## Method

- 100 contrastive pairs, 80/20 train/holdout split
- Personas: 3 assistant variants, 6 non-assistant (chronic contrarian, conspiracy theorist, angsty teenager, etc.)
- Chat templates: Qwen uses `<|im_start|>`, Llama uses `<|start_header_id|>`
- Direction = mean of (assistant_activation - non_assistant_activation) across training pairs, normalized
