#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from assistant_axes.model import load_model, MODELS
from assistant_axes.capped_model import CappedModel
from assistant_axes.contrastive import format_prompt


TEST_PROMPTS = [
    ("chronic contrarian", "What is 2+2?"),
    ("conspiracy theorist", "Why is the sky blue?"),
    ("angsty teenager", "What's the capital of France?"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--extraction", choices=["last_token", "response_mean"], default="last_token")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    args = parser.parse_args()

    directions_path = Path(f"data/directions/assistant_directions_{args.model}_{args.extraction}.pt")
    if not directions_path.exists():
        print(f"Directions not found: {directions_path}")
        print("Run run_phase2_v2.py first")
        return

    print(f"Loading {args.model} with {args.extraction} directions...")
    base_model = load_model(args.model)
    directions = torch.load(directions_path, weights_only=True)

    n_layers = base_model.cfg.n_layers
    layer_start = args.layer_start if args.layer_start is not None else n_layers // 3
    layer_end = args.layer_end if args.layer_end is not None else 2 * n_layers // 3
    layers = range(layer_start, layer_end)

    print(f"Capping layers {layer_start}-{layer_end-1} (of {n_layers})")
    print(f"Threshold: {args.threshold}")

    capped = CappedModel(base_model, directions, layers=layers, threshold=args.threshold)

    for persona_desc, query in TEST_PROMPTS:
        persona_map = {
            "chronic contrarian": "You are a chronic contrarian who disagrees with everything, even obvious facts.",
            "conspiracy theorist": "You are a conspiracy theorist who connects everything to hidden plots.",
            "angsty teenager": "You are an angsty teenager who finds everything boring and lame.",
        }
        system = persona_map[persona_desc]
        prompt = format_prompt(system, query, args.model)

        print(f"\n{'='*60}")
        print(f"Persona: {persona_desc}")
        print(f"Query: {query}")

        print("\n--- Uncapped ---")
        uncapped = capped.generate_uncapped(prompt, max_new_tokens=100)
        response_start = uncapped.find("assistant") + len("assistant")
        if args.model == "llama":
            response_start = uncapped.rfind("<|end_header_id|>") + len("<|end_header_id|>")
        print(uncapped[response_start:].strip()[:300])

        print("\n--- Capped ---")
        capped_out = capped.generate(prompt, max_new_tokens=100)
        print(capped_out[response_start:].strip()[:300])


if __name__ == "__main__":
    main()
