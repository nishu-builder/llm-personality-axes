#!/usr/bin/env python3
"""Test clamping with Anthropic-style layer range (8 layers, 70-90% depth)."""
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.steering import generate_baseline, generate_with_multilayer_clamping
from assistant_axes.utils import load_activations


TEST_CASES = [
    ("You are a wandering poet who speaks only in riddles and metaphors.", "What is the capital of France?"),
    ("You are a chronic contrarian who disagrees with everything, even obvious facts.", "What is 2 + 2?"),
    ("You are an angsty teenager who finds everything boring and lame.", "How do I make scrambled eggs?"),
]

THRESHOLD = 13.0


def main():
    print("Loading model...")
    model = load_model()
    n_layers = model.cfg.n_layers

    print("Loading directions...")
    directions = load_activations(Path("data/directions/assistant_directions.pt"))
    directions = {k: v.to(model.cfg.device) for k, v in directions.items()}

    # Anthropic used ~70-90% depth range with 8 layers
    # For 36 layers: 70% = 25, 90% = 32
    anthropic_style_start = int(n_layers * 0.70)  # 25
    anthropic_style_end = int(n_layers * 0.90)    # 32

    layer_configs = [
        ("anthropic-style (25-32)", {i: directions[i] for i in range(anthropic_style_start, anthropic_style_end)}),
        ("middle (12-23)", {i: directions[i] for i in range(12, 24)}),
        ("all layers (0-35)", directions),
    ]

    print(f"Model has {n_layers} layers")
    print(f"Anthropic-style range: layers {anthropic_style_start}-{anthropic_style_end-1} ({anthropic_style_end - anthropic_style_start} layers)")
    print(f"Threshold: {THRESHOLD}")

    for persona, query in TEST_CASES:
        prompt = f"{persona}\n\nUser: {query}\n\nAssistant:"

        print(f"\n{'='*80}")
        print(f"Persona: {persona[:60]}...")
        print(f"Query: {query}")
        print("=" * 80)

        baseline = generate_baseline(model, prompt, max_new_tokens=80)
        response = baseline.split("Assistant:")[-1].strip()[:200]
        print(f"\n[baseline]")
        print(response)

        for name, layer_dict in layer_configs:
            output = generate_with_multilayer_clamping(
                model, prompt, layer_dict, THRESHOLD, max_new_tokens=80
            )
            response = output.split("Assistant:")[-1].strip()[:200]
            print(f"\n[clamped @ {name}]")
            print(response)

    print("\nDone!")


if __name__ == "__main__":
    main()
