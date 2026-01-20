# Assistant Axes

Replicating Anthropic's [assistant axis](https://www.anthropic.com/research/assistant-axis) work on open-source models.

## What this is

Looking for directions in a model's hidden states that correspond to "being an assistant" vs not. If we find them, we can use them to steer behavior or fingerprint personality from text.

## How it works

Run the same question through the model twice—once with an assistant system prompt, once with something else (angsty teenager, conspiracy theorist, etc). The difference in activations points from "not assistant" toward "assistant."

The model produces activations at every layer and every token position. We grab the last token (where the model has seen everything) at each layer separately. Average a bunch of these diffs together and you get a direction per layer.

## Setup

```bash
# with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# or with pip
pip install -e .
```

Requires gated model access for some models. Run `huggingface-cli login` first.

## Usage

```bash
python scripts/verify_extraction.py      # phase 1: check extraction works
python scripts/run_phase2.py             # phase 2: find the direction
python scripts/run_phase3.py             # phase 3: test additive steering
python scripts/run_clamping.py           # phase 4: test clamping (anthropic approach)
```

Phase 2 outputs:
- `data/directions/assistant_directions.pt` - direction vector per layer
- `data/directions/evaluation_results.pt` - holdout eval metrics

## Findings

### [Phase 2: Direction Discovery](docs/findings/phase2-assistant-axis.md)

Found a clear assistant direction. Layer 25 separates assistant/non-assistant with 95% accuracy (Cohen's d = 7.08). The direction exists.

### [Phase 3: Additive Steering](docs/findings/phase3-steering.md)

Additive steering is fragile. Low scales do nothing visible; high scales cause incoherence before producing clean behavioral shifts. Classification accuracy doesn't predict steering effectiveness.

### [Phase 4: Clamping](docs/findings/phase4-clamping.md)

Anthropic-style clamping at a single layer didn't override explicit system prompt personas at tested thresholds. Preserves normal assistant behavior. Open question whether multi-layer or more aggressive clamping would work.
