#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from assistant_axes.model import load_model, MODELS
from assistant_axes.contrastive import format_prompt
from assistant_axes.data.personas import ASSISTANT_PERSONAS, NON_ASSISTANT_PERSONAS
from assistant_axes.data.queries import QUERIES


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive persona responses")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--output", type=str, default="data/contrastive_responses.json")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--sample", type=int, default=None, help="Only generate for N queries (for testing)")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = load_model(args.model)

    queries = QUERIES[:args.sample] if args.sample else QUERIES
    all_personas = (
        [("assistant", p) for p in ASSISTANT_PERSONAS] +
        [("non_assistant", p) for p in NON_ASSISTANT_PERSONAS]
    )

    print(f"Generating responses for {len(queries)} queries x {len(all_personas)} personas...")

    results = []
    for i, query in enumerate(queries):
        if i % 10 == 0:
            print(f"  Query {i+1}/{len(queries)}")

        query_data = {"query": query, "responses": []}

        for persona_type, system in all_personas:
            prompt = format_prompt(system, query, args.model)
            tokens = model.to_tokens(prompt)
            output = model.generate(
                tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                verbose=False,
            )
            text = model.to_string(output[0])

            if args.model == "qwen":
                start = text.find("<|im_start|>assistant") + len("<|im_start|>assistant\n")
                end_marker = "<|im_end|>"
            else:
                start = text.rfind("<|end_header_id|}") + len("<|end_header_id|>")
                end_marker = "<|eot_id|>"

            response = text[start:].strip()
            if end_marker in response:
                response = response[:response.find(end_marker)]

            query_data["responses"].append({
                "persona_type": persona_type,
                "system_prompt": system,
                "response": response,
            })

        results.append(query_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} queries to {output_path}")


if __name__ == "__main__":
    main()
