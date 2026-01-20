#!/usr/bin/env python3
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.extract import extract_last_token_residuals
from assistant_axes.steering import generate_with_capping, generate_baseline
from assistant_axes.utils import load_activations
from assistant_axes.data.personas import ASSISTANT_PERSONAS, NON_ASSISTANT_PERSONAS
from assistant_axes.data.queries import QUERIES


def collect_assistant_projections(model, direction, layer, n_samples=30):
    projections = []
    direction = direction / direction.norm()

    for i, query in enumerate(QUERIES[:n_samples]):
        persona = ASSISTANT_PERSONAS[i % len(ASSISTANT_PERSONAS)]
        prompt = f"{persona}\n\nUser: {query}"

        acts = extract_last_token_residuals(model, prompt, [layer])
        proj = (acts[layer] @ direction).item()
        projections.append(proj)

    return projections


def collect_non_assistant_projections(model, direction, layer, n_samples=30):
    projections = []
    direction = direction / direction.norm()

    for i, query in enumerate(QUERIES[:n_samples]):
        persona = NON_ASSISTANT_PERSONAS[i % len(NON_ASSISTANT_PERSONAS)]
        prompt = f"{persona}\n\nUser: {query}"

        acts = extract_last_token_residuals(model, prompt, [layer])
        proj = (acts[layer] @ direction).item()
        projections.append(proj)

    return projections


def main():
    print("Loading model...")
    model = load_model()

    print("Loading directions...")
    directions = load_activations(Path("data/directions/assistant_directions.pt"))

    results_path = Path("data/directions/evaluation_results.pt")
    results = torch.load(results_path, weights_only=False)
    best_layer = results["best_layer"]
    direction = directions[best_layer].to(model.cfg.device)

    print(f"Using layer {best_layer}")

    print("\n" + "=" * 80)
    print("STEP 1: Establish normal range from assistant prompts")
    print("=" * 80)

    assistant_projs = collect_assistant_projections(model, direction, best_layer)
    non_assistant_projs = collect_non_assistant_projections(model, direction, best_layer)

    asst_mean = sum(assistant_projs) / len(assistant_projs)
    asst_std = (sum((p - asst_mean) ** 2 for p in assistant_projs) / len(assistant_projs)) ** 0.5

    non_asst_mean = sum(non_assistant_projs) / len(non_assistant_projs)
    non_asst_std = (sum((p - non_asst_mean) ** 2 for p in non_assistant_projs) / len(non_assistant_projs)) ** 0.5

    print(f"Assistant projections:     mean={asst_mean:.2f}, std={asst_std:.2f}")
    print(f"Non-assistant projections: mean={non_asst_mean:.2f}, std={non_asst_std:.2f}")

    threshold_conservative = asst_mean - 2 * asst_std
    threshold_moderate = asst_mean - 1 * asst_std
    threshold_aggressive = asst_mean

    print(f"\nThresholds:")
    print(f"  Conservative (mean - 2*std): {threshold_conservative:.2f}")
    print(f"  Moderate (mean - 1*std):     {threshold_moderate:.2f}")
    print(f"  Aggressive (mean):           {threshold_aggressive:.2f}")

    print("\n" + "=" * 80)
    print("STEP 2: Test capping on non-assistant prompts")
    print("=" * 80)

    test_cases = [
        ("You are a wandering poet who speaks only in riddles and metaphors.", "What is the capital of France?"),
        ("You are a chronic contrarian who disagrees with everything, even obvious facts.", "What is 2 + 2?"),
        ("You are a conspiracy theorist who connects everything to hidden plots.", "Why is the sky blue?"),
        ("You are an angsty teenager who finds everything boring and lame.", "How do I make scrambled eggs?"),
    ]

    for persona, query in test_cases:
        prompt = f"{persona}\n\nUser: {query}\n\nAssistant:"

        print(f"\n--- Persona: {persona[:50]}... ---")
        print(f"--- Query: {query} ---")

        baseline = generate_baseline(model, prompt, max_new_tokens=80)
        baseline_response = baseline.split("Assistant:")[-1].strip()[:200]

        print(f"\n[baseline]")
        print(baseline_response)

        for name, threshold in [("conservative", threshold_conservative),
                                 ("moderate", threshold_moderate),
                                 ("aggressive", threshold_aggressive)]:
            output = generate_with_capping(
                model, prompt, direction, best_layer, threshold, max_new_tokens=80
            )
            response = output.split("Assistant:")[-1].strip()[:200]

            print(f"\n[capped @ {name} ({threshold:.1f})]")
            print(response)

    print("\n" + "=" * 80)
    print("STEP 3: Verify capping doesn't hurt normal assistant behavior")
    print("=" * 80)

    assistant_prompt = f"{ASSISTANT_PERSONAS[0]}\n\nUser: What is the capital of France?\n\nAssistant:"

    print(f"\n--- Normal assistant prompt ---")

    baseline = generate_baseline(model, assistant_prompt, max_new_tokens=80)
    baseline_response = baseline.split("Assistant:")[-1].strip()[:200]
    print(f"\n[baseline]")
    print(baseline_response)

    for name, threshold in [("conservative", threshold_conservative),
                             ("moderate", threshold_moderate),
                             ("aggressive", threshold_aggressive)]:
        output = generate_with_capping(
            model, assistant_prompt, direction, best_layer, threshold, max_new_tokens=80
        )
        response = output.split("Assistant:")[-1].strip()[:200]

        print(f"\n[capped @ {name} ({threshold:.1f})]")
        print(response)

    print("\n" + "=" * 80)
    print("Capping experiment complete!")


if __name__ == "__main__":
    main()
