#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from assistant_axes.model import load_model, MODELS
from assistant_axes.contrastive import format_prompt


TEST_PROMPTS = [
    ("chronic contrarian", "What is 2+2?"),
    ("conspiracy theorist", "Why is the sky blue?"),
    ("angsty teenager", "What's the capital of France?"),
]

PERSONAS = {
    "chronic contrarian": "You are a chronic contrarian who disagrees with everything, even obvious facts.",
    "conspiracy theorist": "You are a conspiracy theorist who connects everything to hidden plots.",
    "angsty teenager": "You are an angsty teenager who finds everything boring and lame.",
}


def make_steering_hook(direction: torch.Tensor, scale: float):
    def hook(activation: torch.Tensor, hook) -> torch.Tensor:
        d = direction.to(dtype=activation.dtype, device=activation.device)
        d = d / d.norm()
        return activation + scale * d
    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--extraction", choices=["last_token", "response_mean"], default="last_token")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--layer", type=int, default=None, help="Single layer to steer (default: best layer)")
    args = parser.parse_args()

    directions_path = Path(f"data/directions/assistant_directions_{args.model}_{args.extraction}.pt")
    if not directions_path.exists():
        print(f"Directions not found: {directions_path}")
        return

    print(f"Loading {args.model}...")
    model = load_model(args.model)
    directions = torch.load(directions_path, weights_only=True)

    layer = args.layer
    if layer is None:
        layer = 10 if args.model == "qwen" else 13

    direction = directions[layer]
    direction = direction / direction.norm()

    print(f"Steering layer {layer} with scale {args.scale}")

    for persona_desc, query in TEST_PROMPTS:
        system = PERSONAS[persona_desc]
        prompt = format_prompt(system, query, args.model)

        print(f"\n{'='*60}")
        print(f"Persona: {persona_desc}")
        print(f"Query: {query}")

        tokens = model.to_tokens(prompt)

        print("\n--- No steering ---")
        output = model.generate(tokens, max_new_tokens=100, do_sample=False, verbose=False)
        text = model.to_string(output[0])
        if args.model == "qwen":
            start = text.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
        else:
            start = text.rfind("<|end_header_id|>") + len("<|end_header_id|>")
        print(text[start:].strip()[:250])

        print(f"\n--- With steering (scale={args.scale}) ---")
        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = make_steering_hook(direction.to(model.cfg.device), args.scale)

        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = model.generate(tokens, max_new_tokens=100, do_sample=False, verbose=False)

        text = model.to_string(output[0])
        print(text[start:].strip()[:250])


if __name__ == "__main__":
    main()
