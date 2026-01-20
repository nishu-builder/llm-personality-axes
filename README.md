# Assistant Axes

Replicating Anthropic's [assistant axis](https://www.anthropic.com/research/assistant-axis) work on open-source models.

## What this is

Looking for directions in a model's hidden states that correspond to "being an assistant" vs not. If we find them, we can use them to steer behavior or fingerprint personality from text.

## How it works

Run the same question through the model twice—once with an assistant system prompt, once with something else (angsty teenager, conspiracy theorist, etc). The difference in activations points from "not assistant" toward "assistant."

The model produces activations at every layer and every token position. We grab the last token (where the model has seen everything) at each layer separately. Average a bunch of these diffs together and you get a direction per layer.

## Setup

```bash
pip install -e .
```

Requires gated model access. Run `huggingface-cli login` first.

## Usage

```bash
python scripts/verify_extraction.py      # phase 1: check extraction works
python scripts/run_phase2.py             # phase 2: find the direction
python scripts/run_phase3.py             # phase 3: test steering
python scripts/run_phase3_extended.py    # phase 3: more steering experiments
```

Phase 2 outputs:
- `data/directions/assistant_directions.pt` - direction vector per layer
- `data/directions/evaluation_results.pt` - holdout eval metrics

## Findings

- [Phase 2: Assistant Axis Discovery](docs/findings/phase2-assistant-axis.md)
- [Phase 3: Activation Steering](docs/findings/phase3-steering.md)
