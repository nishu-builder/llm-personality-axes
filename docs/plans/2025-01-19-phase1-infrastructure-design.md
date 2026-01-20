# Phase 1: Infrastructure Design

## Goal

Set up Llama 3.2 3B Instruct with TransformerLens for activation extraction. Build a pipeline that takes prompts, runs them through the model, and extracts residual stream activations at each layer.

## Hardware

- Development: MacBook Pro (MPS)
- Training/experiments: AWS g6.2xlarge (L4 GPU, 24GB VRAM)

## Project Structure

```
assistant-axes/
├── pyproject.toml
├── src/
│   └── assistant_axes/
│       ├── __init__.py
│       ├── model.py        # Model loading
│       ├── extract.py      # Activation extraction
│       └── utils.py        # Tensor I/O, device helpers
├── scripts/
│   └── verify_extraction.py
├── data/
│   └── activations/
└── docs/
    └── plans/
```

## Dependencies

- transformer-lens>=2.0
- torch>=2.0
- transformers, accelerate, safetensors
- einops, jaxtyping

## Core Components

### model.py

Load Llama 3.2 3B Instruct via TransformerLens HookedTransformer. Handle device placement (CUDA/MPS). Model config: n_layers=28, d_model=3072.

### extract.py

Extract residual stream activations using `run_with_cache`. TransformerLens hook names: `blocks.{i}.hook_resid_post` for post-layer residuals. Returns dict mapping layer index to tensor of shape (seq_len, d_model).

### verify_extraction.py

Test script to validate the pipeline works:
1. Load model
2. Run test prompts (with/without system prompts)
3. Verify activation shapes match expected dimensions
4. Save example activations to disk

## Success Criteria

- All prompts return activations with shape (seq_len, 3072) per layer
- 28 layers extracted
- No OOM on L4 or MacBook
- Activations can be saved/loaded from disk
