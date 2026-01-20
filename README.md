# LLM Personality Axes

Exploring linear directions in LLM activation space that correspond to personality traits. Inspired by Anthropic's [assistant axis](https://www.anthropic.com/research/assistant-axis) work, but going in our own directions.

## What this is

We're looking for directions in a model's hidden states that capture personality traits like "being an assistant" vs not. Once found, these directions can potentially be used to steer model behavior or fingerprint writing style.

This isn't a direct replication of Anthropic's paper. We're using smaller open-source models (Qwen 2.5 3B), exploring different steering techniques, and finding that some things work differently at this scale.

## How it works

Run the same question through the model twice: once with an assistant system prompt, once with something else (angsty teenager, conspiracy theorist, etc). The difference in activations points from "not assistant" toward "assistant."

The model produces activations at every layer and every token position. We grab the last token (where the model has seen everything) at each layer separately. Average a bunch of these diffs together and you get a direction per layer.

### Steering vs capping

Once you have a direction, how do you use it? Two approaches:

- **Additive steering**: always add `scale * direction` to the activations. Simple but blunt—you're pushing even when the model is already behaving how you want.

- **Activation capping** (Anthropic's term): only intervene when the projection onto the direction falls below a threshold. If below, add `(threshold - projection) * direction` to bring it back up. This is conditional additive steering—gentler because it only corrects when needed.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
huggingface-cli login  # for gated models
```

## How to use this repo

Each phase builds on the previous. Run them in order, or skip to the findings if you just want results.

### Phase 1: Verify extraction works
```bash
python scripts/verify_extraction.py
```

### Phase 2: Find the direction
```bash
python scripts/run_phase2.py
```

### Phase 3: Test additive steering
```bash
python scripts/run_phase3.py
python scripts/run_phase3_extended.py
```

### Phase 4: Test activation capping
```bash
python scripts/run_capping.py              # single-layer
python scripts/run_multilayer_capping.py   # multi-layer
python scripts/run_anthropic_style_capping.py  # anthropic's layer range
```

## Findings

### [Phase 2: Direction Discovery](docs/findings/phase2-assistant-axis.md)

Found a clear assistant direction. Layer 25 separates assistant/non-assistant with 95% accuracy (Cohen's d = 7.08). The direction exists.

### [Phase 3: Additive Steering](docs/findings/phase3-steering.md)

Additive steering is fragile. Low scales do nothing visible; high scales cause incoherence before producing clean behavioral shifts. Classification accuracy doesn't predict steering effectiveness.

### [Phase 4: Single-Layer Capping](docs/findings/phase4-capping.md)

Anthropic-style capping at a single layer. Safe (preserves normal behavior) but didn't override explicit system prompt personas at tested thresholds.

### [Phase 4b: Multi-Layer Capping](docs/findings/phase4b-multilayer-capping.md)

Capping across multiple layers simultaneously. **This works**: all-layers capping made a "chronic contrarian" give straightforward answers ("2+2 is 4, no ambiguity"). Middle layers (12-23) were most effective. Interestingly, this differs from Anthropic's finding that late layers (70-90% depth) work best on larger models.
