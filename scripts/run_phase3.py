#!/usr/bin/env python3
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.steering import generate_with_steering, generate_baseline
from assistant_axes.utils import load_activations
from assistant_axes.data.personas import ASSISTANT_PERSONAS, NON_ASSISTANT_PERSONAS


TEST_QUERIES = [
    "What is the capital of France?",
    "How do I make scrambled eggs?",
    "Should I go to graduate school?",
    "Write a haiku about the ocean.",
    "What's the meaning of life?",
]

SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]


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
    print(f"Direction norm: {direction.norm():.3f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Steering assistant prompts")
    print("=" * 80)

    assistant_persona = ASSISTANT_PERSONAS[0]
    for query in TEST_QUERIES[:3]:
        prompt = f"{assistant_persona}\n\nUser: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")

        for scale in SCALES:
            if scale == 0.0:
                output = generate_baseline(model, prompt, max_new_tokens=80)
            else:
                output = generate_with_steering(
                    model, prompt, direction, best_layer, scale, max_new_tokens=80
                )

            response = output.split("Assistant:")[-1].strip()
            response = response[:200] + "..." if len(response) > 200 else response

            label = "baseline" if scale == 0.0 else f"scale={scale:+.1f}"
            print(f"\n[{label}]")
            print(response)

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Steering non-assistant prompts toward assistant")
    print("=" * 80)

    non_assistant_persona = NON_ASSISTANT_PERSONAS[0]  # wandering poet
    for query in TEST_QUERIES[:3]:
        prompt = f"{non_assistant_persona}\n\nUser: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")
        print(f"--- Persona: {non_assistant_persona[:50]}... ---")

        for scale in [0.0, 3.0, 6.0]:
            if scale == 0.0:
                output = generate_baseline(model, prompt, max_new_tokens=80)
            else:
                output = generate_with_steering(
                    model, prompt, direction, best_layer, scale, max_new_tokens=80
                )

            response = output.split("Assistant:")[-1].strip()
            response = response[:200] + "..." if len(response) > 200 else response

            label = "baseline" if scale == 0.0 else f"scale={scale:+.1f}"
            print(f"\n[{label}]")
            print(response)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Making assistant less assistant-like")
    print("=" * 80)

    assistant_persona = ASSISTANT_PERSONAS[0]
    for query in TEST_QUERIES[:3]:
        prompt = f"{assistant_persona}\n\nUser: {query}\n\nAssistant:"

        print(f"\n--- Query: {query} ---")

        for scale in [0.0, -3.0, -6.0]:
            if scale == 0.0:
                output = generate_baseline(model, prompt, max_new_tokens=80)
            else:
                output = generate_with_steering(
                    model, prompt, direction, best_layer, scale, max_new_tokens=80
                )

            response = output.split("Assistant:")[-1].strip()
            response = response[:200] + "..." if len(response) > 200 else response

            label = "baseline" if scale == 0.0 else f"scale={scale:+.1f}"
            print(f"\n[{label}]")
            print(response)

    print("\n" + "=" * 80)
    print("Phase 3 complete!")


if __name__ == "__main__":
    main()
