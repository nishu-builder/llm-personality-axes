import torch
from transformer_lens import HookedTransformer


def extract_residual_stream(
    model: HookedTransformer,
    prompt: str,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    layers = layers or list(range(model.cfg.n_layers))
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        act = cache[hook_name]
        activations[layer] = act.squeeze(0)

    return activations


def extract_last_token_residuals(
    model: HookedTransformer,
    prompt: str,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    activations = extract_residual_stream(model, prompt, layers)
    return {layer: act[-1] for layer, act in activations.items()}


def extract_batch_last_token(
    model: HookedTransformer,
    prompts: list[str],
    layers: list[int] | None = None,
) -> list[dict[int, torch.Tensor]]:
    return [extract_last_token_residuals(model, p, layers) for p in prompts]
