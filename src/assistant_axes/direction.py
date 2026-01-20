import torch
from transformer_lens import HookedTransformer

from assistant_axes.extract import extract_last_token_residuals


def project_onto_direction(
    activation: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    direction = direction / direction.norm()
    return (activation @ direction).item()


def evaluate_separation(
    model: HookedTransformer,
    direction: torch.Tensor,
    layer: int,
    assistant_prompts: list[str],
    non_assistant_prompts: list[str],
) -> dict:
    assistant_projections = []
    non_assistant_projections = []

    for prompt in assistant_prompts:
        acts = extract_last_token_residuals(model, prompt, [layer])
        proj = project_onto_direction(acts[layer], direction)
        assistant_projections.append(proj)

    for prompt in non_assistant_prompts:
        acts = extract_last_token_residuals(model, prompt, [layer])
        proj = project_onto_direction(acts[layer], direction)
        non_assistant_projections.append(proj)

    assistant_mean = sum(assistant_projections) / len(assistant_projections)
    non_assistant_mean = sum(non_assistant_projections) / len(non_assistant_projections)

    assistant_std = (sum((p - assistant_mean) ** 2 for p in assistant_projections) / len(assistant_projections)) ** 0.5
    non_assistant_std = (sum((p - non_assistant_mean) ** 2 for p in non_assistant_projections) / len(non_assistant_projections)) ** 0.5

    pooled_std = ((assistant_std ** 2 + non_assistant_std ** 2) / 2) ** 0.5
    cohens_d = (assistant_mean - non_assistant_mean) / pooled_std if pooled_std > 0 else 0

    correct = sum(1 for p in assistant_projections if p > 0) + sum(1 for p in non_assistant_projections if p < 0)
    accuracy = correct / (len(assistant_projections) + len(non_assistant_projections))

    return {
        "assistant_mean": assistant_mean,
        "non_assistant_mean": non_assistant_mean,
        "assistant_std": assistant_std,
        "non_assistant_std": non_assistant_std,
        "cohens_d": cohens_d,
        "accuracy": accuracy,
    }


def find_best_layer(
    model: HookedTransformer,
    directions: dict[int, torch.Tensor],
    assistant_prompts: list[str],
    non_assistant_prompts: list[str],
    verbose: bool = True,
) -> tuple[int, dict]:
    results = {}

    for layer, direction in directions.items():
        if verbose:
            print(f"Evaluating layer {layer}...")

        metrics = evaluate_separation(
            model, direction, layer, assistant_prompts, non_assistant_prompts
        )
        results[layer] = metrics

        if verbose:
            print(f"  Cohen's d: {metrics['cohens_d']:.3f}, Accuracy: {metrics['accuracy']:.3f}")

    best_layer = max(results.keys(), key=lambda l: abs(results[l]["cohens_d"]))

    return best_layer, results
