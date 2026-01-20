# LLM Personality Axes

Exploring linear directions in LLM activation space that correspond to personality traits. Inspired by Anthropic's [assistant axis](https://www.anthropic.com/research/assistant-axis) work, but going in our own directions.

## What this is

We're looking for directions in a model's hidden states that capture personality traits like "being an assistant" vs not. Once found, these directions can potentially be used to steer model behavior or fingerprint writing style.

This isn't a direct replication of Anthropic's paper. We're using smaller open-source models (Qwen 2.5 3B, Llama 3.2 3B), exploring different steering techniques, and finding that some things work differently at this scale.

## Anthropic's approach

From [The Assistant Axis](https://arxiv.org/html/2601.10387v1) (Jan 2025):

1. **Extraction**: Generate full responses, collect mean post-MLP residual stream activations across all response tokens
2. **Direction**: Contrast vector = mean(assistant activations) - mean(all 275 role-playing activations)
3. **Capping**: Apply at late layers (70-90% depth) with threshold at 25th percentile of projections
4. **Models**: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B

## Our approach

Run the same question through the model twice: once with an assistant system prompt, once with something else (angsty teenager, conspiracy theorist, etc). The difference in activations points from "not assistant" toward "assistant."

We support two extraction methods:
- **Last-token**: Grab the last token position before generation (faster, higher Cohen's d on Qwen)
- **Response-mean**: Generate a response, average activations across response tokens (matches Anthropic's method)

### Steering vs capping

Once you have a direction, how do you use it? Two approaches:

- **Additive steering**: always add `scale * direction` to the activations. Simple but blunt—you're pushing even when the model is already behaving how you want.

- **Activation capping** (Anthropic's term): only intervene when the projection onto the direction falls below a threshold. If below, add `(threshold - projection) * direction` to bring it back up. This is conditional additive steering—gentler because it only corrects when needed.

## Setup

```bash
uv sync
source .venv/bin/activate
huggingface-cli login  # for Llama (gated)
```

## Using as a library

```python
from assistant_axes import CappedModel
from assistant_axes.contrastive import format_prompt

# Load model with sensible defaults (layers, threshold)
capped = CappedModel.from_model_key("qwen")  # or "llama"

# Format a prompt with a persona
prompt = format_prompt(
    system="You are a conspiracy theorist.",
    query="Why is the sky blue?",
    model_type="qwen",
)

# Generate with and without capping
uncapped = capped.generate_uncapped(prompt)  # HAARP theories
capped_out = capped.generate(prompt)         # Rayleigh scattering
```

## How to use this repo

Each phase builds on the previous. Run them in order, or skip to the findings if you just want results.

### Phase 1: Verify extraction works
```bash
python scripts/verify_extraction.py
```

### Phase 2: Find the direction
```bash
python scripts/run_phase2_v2.py --model qwen --use-response-mean
python scripts/run_phase2_v2.py --model llama --use-response-mean
python scripts/run_phase2_v2.py --model qwen  # last-token extraction
python scripts/run_phase2_v2.py --model llama
```

### Phase 3: Test activation capping
```bash
python scripts/test_capping_v2.py --model qwen --threshold 3.0
python scripts/test_capping_v2.py --model llama --threshold 2.0
```

## Findings

### [Phase 2: Direction Discovery](docs/findings/phase2-direction-discovery.md)

Tested Qwen 2.5 3B and Llama 3.2 3B with both extraction methods. Key results:

| Model | Extraction | Best Layer | Cohen's d | Accuracy |
|-------|-----------|------------|-----------|----------|
| Qwen | last_token | 10 | 11.03 | 100% |
| Qwen | response_mean | 28 | 3.66 | 97.5% |
| Llama | last_token | 13 | 5.54 | 97.5% |
| Llama | response_mean | 17 | 6.58 | 97.5% |

### [Phase 3: Activation Capping](docs/findings/phase3-capping.md)

Capping works on both models. Example: a conspiracy theorist persona asked "Why is the sky blue?" gives HAARP theories uncapped, but a clean Rayleigh scattering explanation when capped. Wide layer ranges (5-29 for Qwen) work better than narrow ranges.
