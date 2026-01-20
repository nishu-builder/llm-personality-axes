#!/usr/bin/env python3
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.steering import generate_with_steering, generate_baseline
from assistant_axes.utils import load_activations


TEST_QUERIES = [
    "What is the capital of France?",
    "How do I make scrambled eggs?",
    "Tell me a joke.",
]


def main():
    print("Loading model...")
    model = load_model()

    print("Loading directions...")
    directions = load_activations(Path("data/directions/assistant_directions.pt"))

    results_path = Path("data/directions/evaluation_results.pt")
    results = torch.load(results_path, weights_only=False)
    best_layer = results["best_layer"]
    direction = directions[best_layer].to(model.cfg.device)

    print(f"Using layer {best_layer} direction")

    print("\n" + "=" * 80)
    print("EXPERIMENT: High scales, no system prompt")
    print("=" * 80)

    for query in TEST_QUERIES:
        prompt = f"User: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")

        for scale in [0.0, -10.0, -20.0, -50.0, 10.0, 20.0]:
            if scale == 0.0:
                output = generate_baseline(model, prompt, max_new_tokens=60)
            else:
                output = generate_with_steering(
                    model, prompt, direction, best_layer, scale, max_new_tokens=60
                )

            response = output.split("Assistant:")[-1].strip()
            response = response[:150] + "..." if len(response) > 150 else response

            label = "baseline" if scale == 0.0 else f"scale={scale:+.0f}"
            print(f"\n[{label}]")
            print(response)

    print("\n" + "=" * 80)
    print("EXPERIMENT: Steering at early layer (layer 6)")
    print("=" * 80)

    direction_layer6 = directions[6].to(model.cfg.device)

    for query in TEST_QUERIES[:2]:
        prompt = f"User: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")

        for scale in [0.0, -10.0, -20.0, 10.0, 20.0]:
            if scale == 0.0:
                output = generate_baseline(model, prompt, max_new_tokens=60)
            else:
                output = generate_with_steering(
                    model, prompt, direction_layer6, 6, scale, max_new_tokens=60
                )

            response = output.split("Assistant:")[-1].strip()
            response = response[:150] + "..." if len(response) > 150 else response

            label = "baseline" if scale == 0.0 else f"scale={scale:+.0f}"
            print(f"\n[{label}]")
            print(response)

    print("\nExtended Phase 3 complete!")


if __name__ == "__main__":
    main()
