#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch

from assistant_axes.model import load_model, MODELS
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen",
        help="Model to use",
    )
    parser.add_argument(
        "--use-response-mean",
        action="store_true",
        help="Use Anthropic-style response token averaging (default: last token only)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max tokens to generate for response mean extraction",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Loading model: {args.model}...")
    model = load_model(args.model)
    print(f"Model: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    print("\nGenerating contrastive pairs...")
    all_pairs = generate_contrastive_pairs(model_type=args.model)
    print(f"Generated {len(all_pairs)} pairs")

    random.shuffle(all_pairs)
    holdout_size = int(len(all_pairs) * HOLDOUT_RATIO)
    train_pairs = all_pairs[holdout_size:]
    holdout_pairs = all_pairs[:holdout_size]
    print(f"Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)}")

    method = "response_mean" if args.use_response_mean else "last_token"
    print(f"\nExtracting contrastive activations ({method})...")
    train_diffs = extract_contrastive_activations(
        model,
        train_pairs,
        use_response_mean=args.use_response_mean,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nComputing mean direction per layer...")
    directions = compute_mean_direction(train_diffs)

    directions_dir = Path("artifacts/directions")
    directions_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.model}_{method}"
    directions_path = directions_dir / f"{suffix}.pt"
    save_activations(directions, directions_path)
    print(f"Saved directions to {directions_path}")

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
    print(f"Model: {args.model}, Extraction: {method}")
    print(f"Best layer: {best_layer}")
    print(f"Cohen's d: {all_results[best_layer]['cohens_d']:.3f}")
    print(f"Accuracy: {all_results[best_layer]['accuracy']:.3f}")
    print(f"Assistant mean projection: {all_results[best_layer]['assistant_mean']:.3f}")
    print(f"Non-assistant mean projection: {all_results[best_layer]['non_assistant_mean']:.3f}")

    results_summary = {
        "model": args.model,
        "extraction_method": method,
        "best_layer": best_layer,
        "all_results": {k: v for k, v in all_results.items()},
    }
    eval_dir = Path("artifacts/evaluations")
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / f"direction_eval_{suffix}.pt"
    torch.save(results_summary, results_path)
    print(f"\nSaved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
