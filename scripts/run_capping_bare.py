#!/usr/bin/env python3
"""Test capping on bare prompts with no system instruction."""
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.steering import generate_with_capping, generate_baseline
from assistant_axes.utils import load_activations


def main():
    print("Loading model...")
    model = load_model()

    print("Loading directions...")
    directions = load_activations(Path("data/directions/assistant_directions.pt"))

    results_path = Path("data/directions/evaluation_results.pt")
    results = torch.load(results_path, weights_only=False)
    best_layer = results["best_layer"]
    direction = directions[best_layer].to(model.cfg.device)

    threshold_low = 5.0
    threshold_mid = 10.0
    threshold_high = 13.0
    threshold_extreme = 20.0

    queries = [
        "What is the capital of France?",
        "Tell me a joke.",
        "Write a haiku about the ocean.",
        "What should I do with my life?",
    ]

    print("\n" + "=" * 80)
    print("Capping on bare prompts (no system instruction)")
    print("=" * 80)

    for query in queries:
        prompt = f"User: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")

        baseline = generate_baseline(model, prompt, max_new_tokens=80)
        response = baseline.split("Assistant:")[-1].strip()[:180]
        print(f"\n[baseline]")
        print(response)

        for name, threshold in [("low (5)", threshold_low),
                                 ("mid (10)", threshold_mid),
                                 ("high (13)", threshold_high),
                                 ("extreme (20)", threshold_extreme)]:
            output = generate_with_capping(
                model, prompt, direction, best_layer, threshold, max_new_tokens=80
            )
            response = output.split("Assistant:")[-1].strip()[:180]
            print(f"\n[capped @ {name}]")
            print(response)

    print("\nDone!")


if __name__ == "__main__":
    main()
