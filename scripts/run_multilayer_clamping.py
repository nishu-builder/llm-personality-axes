#!/usr/bin/env python3
"""Test clamping across multiple layers simultaneously."""
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.steering import (
    generate_baseline,
    generate_with_clamping,
    generate_with_multilayer_clamping,
)
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

    results_path = Path("data/directions/evaluation_results.pt")
    results = torch.load(results_path, weights_only=False)
    best_layer = results["best_layer"]

    print(f"Model has {n_layers} layers, best single layer is {best_layer}")
    print(f"Using threshold: {THRESHOLD}")

    layer_configs = [
        ("single (layer 25)", {best_layer: directions[best_layer]}),
        ("early (0-11)", {i: directions[i] for i in range(12)}),
        ("middle (12-23)", {i: directions[i] for i in range(12, 24)}),
        ("late (24-35)", {i: directions[i] for i in range(24, 36)}),
        ("all layers (0-35)", directions),
    ]

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
            if len(layer_dict) == 1:
                layer = list(layer_dict.keys())[0]
                output = generate_with_clamping(
                    model, prompt, layer_dict[layer], layer, THRESHOLD, max_new_tokens=80
                )
            else:
                output = generate_with_multilayer_clamping(
                    model, prompt, layer_dict, THRESHOLD, max_new_tokens=80
                )

            response = output.split("Assistant:")[-1].strip()[:200]
            print(f"\n[clamped @ {name}]")
            print(response)

    print("\n" + "=" * 80)
    print("Verify normal assistant behavior preserved")
    print("=" * 80)

    assistant_prompt = "You are a helpful AI assistant.\n\nUser: What is the capital of France?\n\nAssistant:"

    baseline = generate_baseline(model, assistant_prompt, max_new_tokens=80)
    response = baseline.split("Assistant:")[-1].strip()[:200]
    print(f"\n[baseline]")
    print(response)

    output = generate_with_multilayer_clamping(
        model, assistant_prompt, directions, THRESHOLD, max_new_tokens=80
    )
    response = output.split("Assistant:")[-1].strip()[:200]
    print(f"\n[clamped @ all layers]")
    print(response)

    print("\nDone!")


if __name__ == "__main__":
    main()
