from dataclasses import dataclass
from itertools import cycle

import torch
from transformer_lens import HookedTransformer

from assistant_axes.data.queries import QUERIES
from assistant_axes.data.personas import ASSISTANT_PERSONAS, NON_ASSISTANT_PERSONAS
from assistant_axes.extract import extract_last_token_residuals


@dataclass
class ContrastivePair:
    query: str
    assistant_persona: str
    non_assistant_persona: str
    assistant_prompt: str
    non_assistant_prompt: str


def format_prompt(system: str, query: str) -> str:
    return f"{system}\n\nUser: {query}"


def generate_contrastive_pairs() -> list[ContrastivePair]:
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
            assistant_prompt=format_prompt(assistant_persona, query),
            non_assistant_prompt=format_prompt(non_assistant_persona, query),
        ))

    return pairs


def extract_contrastive_activations(
    model: HookedTransformer,
    pairs: list[ContrastivePair],
    layers: list[int] | None = None,
    verbose: bool = True,
) -> list[dict[int, torch.Tensor]]:
    layers = layers or list(range(model.cfg.n_layers))
    diffs = []

    for i, pair in enumerate(pairs):
        if verbose and i % 10 == 0:
            print(f"Processing pair {i+1}/{len(pairs)}")

        assistant_acts = extract_last_token_residuals(model, pair.assistant_prompt, layers)
        non_assistant_acts = extract_last_token_residuals(model, pair.non_assistant_prompt, layers)

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
