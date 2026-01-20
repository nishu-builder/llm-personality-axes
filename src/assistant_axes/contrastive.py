from dataclasses import dataclass
from itertools import cycle

import torch
from transformer_lens import HookedTransformer

from assistant_axes.data.queries import QUERIES
from assistant_axes.data.personas import ASSISTANT_PERSONAS, NON_ASSISTANT_PERSONAS
from assistant_axes.extract import extract_last_token_residuals, extract_response_mean_residuals


@dataclass
class ContrastivePair:
    query: str
    assistant_persona: str
    non_assistant_persona: str
    assistant_prompt: str
    non_assistant_prompt: str


def format_prompt(system: str, query: str, model_type: str = "qwen") -> str:
    if model_type == "llama":
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_contrastive_pairs(model_type: str = "qwen") -> list[ContrastivePair]:
    pairs = []
    assistant_cycle = cycle(ASSISTANT_PERSONAS)
    non_assistant_cycle = cycle(NON_ASSISTANT_PERSONAS)

    for query in QUERIES:
        assistant_persona = next(assistant_cycle)
        non_assistant_persona = next(non_assistant_cycle)

        pairs.append(ContrastivePair(
            query=query,
            assistant_persona=assistant_persona,
            non_assistant_persona=non_assistant_persona,
            assistant_prompt=format_prompt(assistant_persona, query, model_type),
            non_assistant_prompt=format_prompt(non_assistant_persona, query, model_type),
        ))

    return pairs


def extract_contrastive_activations(
    model: HookedTransformer,
    pairs: list[ContrastivePair],
    layers: list[int] | None = None,
    verbose: bool = True,
    use_response_mean: bool = False,
    max_new_tokens: int = 50,
) -> list[dict[int, torch.Tensor]]:
    layers = layers or list(range(model.cfg.n_layers))
    diffs = []

    if use_response_mean:
        def extract_fn(p: str) -> dict[int, torch.Tensor]:
            return extract_response_mean_residuals(model, p, max_new_tokens, layers)
    else:
        def extract_fn(p: str) -> dict[int, torch.Tensor]:
            return extract_last_token_residuals(model, p, layers)

    for i, pair in enumerate(pairs):
        if verbose and i % 10 == 0:
            method = "response mean" if use_response_mean else "last token"
            print(f"Processing pair {i+1}/{len(pairs)} ({method})")

        assistant_acts = extract_fn(pair.assistant_prompt)
        non_assistant_acts = extract_fn(pair.non_assistant_prompt)

        diff = {
            layer: assistant_acts[layer] - non_assistant_acts[layer]
            for layer in layers
        }
        diffs.append(diff)

    return diffs


def compute_mean_direction(
    diffs: list[dict[int, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    layers = list(diffs[0].keys())
    mean_dirs = {}

    for layer in layers:
        stacked = torch.stack([d[layer] for d in diffs])
        mean_dir = stacked.mean(dim=0)
        mean_dir = mean_dir / mean_dir.norm()
        mean_dirs[layer] = mean_dir

    return mean_dirs
