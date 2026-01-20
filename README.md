# Assistant Axes

Replicating and extending Anthropic's [assistant axis](https://www.anthropic.com/research/assistant-axis) interpretability research on open-source models.

## Goal

Find linear directions in activation space that correspond to personality traits, validate they're steerable, then build toward inferring "personality fingerprints" from text samples.

## Setup

```bash
pip install -e .
```

Requires access to gated models (e.g., Llama). Run `huggingface-cli login` and accept model licenses on HuggingFace.

## Usage

### Phase 1: Verify activation extraction
```bash
python scripts/verify_extraction.py
```

### Phase 2: Extract assistant direction
```bash
python scripts/run_phase2.py
```

Outputs:
- `data/directions/assistant_directions.pt` - direction vector per layer
- `data/directions/evaluation_results.pt` - holdout evaluation metrics

### Phase 3: Validate steering
```bash
python scripts/run_phase3.py
python scripts/run_phase3_extended.py  # high scales, layer comparison
```

## Project Structure

```
src/assistant_axes/
├── model.py          # Model loading (TransformerLens)
├── extract.py        # Activation extraction
├── contrastive.py    # Contrastive pair generation
├── direction.py      # Direction computation and evaluation
├── data/
│   ├── queries.py    # 100 diverse test queries
│   └── personas.py   # Assistant and non-assistant personas
└── utils.py          # I/O helpers

scripts/
├── verify_extraction.py   # Phase 1 verification
├── run_phase2.py          # Phase 2 pipeline
├── run_phase3.py          # Phase 3 steering
└── run_phase3_extended.py # Extended steering experiments

docs/
├── plans/            # Design documents
└── findings/         # Experimental results
```

## Findings

- [Phase 2: Assistant Axis Discovery](docs/findings/phase2-assistant-axis.md)
- [Phase 3: Activation Steering](docs/findings/phase3-steering.md)
