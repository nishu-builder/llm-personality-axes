#!/usr/bin/env python3
import random
from pathlib import Path

import torch

from assistant_axes.model import load_model
from assistant_axes.contrastive import (
    generate_contrastive_pairs,
    extract_contrastive_activations,
    compute_mean_direction,
)
from assistant_axes.direction import find_best_layer
from assistant_axes.utils import save_activations


HOLDOUT_RATIO = 0.2
SEED = 42


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading model...")
    model = load_model()
    print(f"Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    print("\nGenerating contrastive pairs...")
    all_pairs = generate_contrastive_pairs()
    print(f"Generated {len(all_pairs)} pairs")

    random.shuffle(all_pairs)
    holdout_size = int(len(all_pairs) * HOLDOUT_RATIO)
    train_pairs = all_pairs[holdout_size:]
    holdout_pairs = all_pairs[:holdout_size]
    print(f"Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)}")

    print("\nExtracting contrastive activations (train set)...")
    train_diffs = extract_contrastive_activations(model, train_pairs)

    print("\nComputing mean direction per layer...")
    directions = compute_mean_direction(train_diffs)

    output_dir = Path("data/directions")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_activations(directions, output_dir / "assistant_directions.pt")
    print(f"Saved directions to {output_dir / 'assistant_directions.pt'}")

    print("\nEvaluating on holdout set...")
    holdout_assistant_prompts = [p.assistant_prompt for p in holdout_pairs]
    holdout_non_assistant_prompts = [p.non_assistant_prompt for p in holdout_pairs]

    best_layer, all_results = find_best_layer(
        model,
        directions,
        holdout_assistant_prompts,
        holdout_non_assistant_prompts,
    )

    print(f"\n{'='*50}")
    print(f"Best layer: {best_layer}")
    print(f"Cohen's d: {all_results[best_layer]['cohens_d']:.3f}")
    print(f"Accuracy: {all_results[best_layer]['accuracy']:.3f}")
    print(f"Assistant mean projection: {all_results[best_layer]['assistant_mean']:.3f}")
    print(f"Non-assistant mean projection: {all_results[best_layer]['non_assistant_mean']:.3f}")

    results_summary = {
        "best_layer": best_layer,
        "all_results": {k: v for k, v in all_results.items()},
    }
    torch.save(results_summary, output_dir / "evaluation_results.pt")
    print(f"\nSaved evaluation results to {output_dir / 'evaluation_results.pt'}")

    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()
