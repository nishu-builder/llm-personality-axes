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

## Scripts

```bash
# 1. Verify extraction works
python scripts/verify_extraction.py

# 2. Find the personality direction
python scripts/find_direction.py --model qwen
python scripts/find_direction.py --model llama
python scripts/find_direction.py --model qwen --use-response-mean  # Anthropic-style

# 3. Test additive steering
python scripts/test_steering.py --model qwen --scale 2.0

# 4. Test activation capping
python scripts/test_capping.py --model qwen --threshold 3.0
python scripts/test_capping.py --model llama --threshold 2.0
```

## Findings

### [Direction Discovery](docs/findings/direction-discovery.md)

We find directions that separate assistant from non-assistant activations on holdout data:

| Model | Extraction | Best Layer | Cohen's d | Accuracy |
|-------|-----------|------------|-----------|----------|
| Qwen | last_token | 10 | 11.03 | 100% |
| Qwen | response_mean | 28 | 3.66 | 97.5% |
| Llama | last_token | 13 | 5.54 | 97.5% |
| Llama | response_mean | 17 | 6.58 | 97.5% |

### [Additive Steering](docs/findings/additive-steering.md)

Additive steering shows a narrow effective range on our 3B models. Low scales (< 5) have minimal visible effect; high scales (> 50) cause incoherence. Effects are inconsistent across personas.

### [Activation Capping](docs/findings/activation-capping.md)

Capping appeared more consistent in our limited testing. Example: a conspiracy theorist asked "Why is the sky blue?" gives HAARP theories uncapped, but a Rayleigh scattering explanation when capped (on Qwen with layers 5-29, threshold 3.0). Results varied by model and persona.
