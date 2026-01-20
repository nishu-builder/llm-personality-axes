#!/usr/bin/env python3
from pathlib import Path

from assistant_axes.model import load_model, get_device
from assistant_axes.extract import extract_residual_stream, extract_last_token_residuals
from assistant_axes.utils import save_activations, load_activations


TEST_PROMPTS = [
    "You are a helpful assistant.\n\nUser: What is 2+2?",
    "You are a rude human who hates questions.\n\nUser: What is 2+2?",
    "What is the capital of France?",
]


def main():
    print(f"Device: {get_device()}")
    print("Loading model...")
    model = load_model()
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Text: {prompt[:50]}...")

        activations = extract_residual_stream(model, prompt)
        print(f"Extracted {len(activations)} layers")

        for layer_idx in [0, model.cfg.n_layers // 2, model.cfg.n_layers - 1]:
            act = activations[layer_idx]
            print(f"  Layer {layer_idx}: shape={tuple(act.shape)}, dtype={act.dtype}")

        last_token = extract_last_token_residuals(model, prompt)
        sample_vec = last_token[model.cfg.n_layers - 1]
        print(f"Last token residual (final layer): shape={tuple(sample_vec.shape)}")

    print("\n--- Testing save/load ---")
    test_path = Path("data/activations/test_activation.pt")
    save_activations(activations, test_path)
    loaded = load_activations(test_path)
    assert len(loaded) == len(activations)
    print(f"Saved and loaded activations from {test_path}")

    print("\nPhase 1 verification complete!")


if __name__ == "__main__":
    main()
