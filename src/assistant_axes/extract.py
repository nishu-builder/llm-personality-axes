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


def extract_response_mean_residuals(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 50,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    layers = layers or list(range(model.cfg.n_layers))
    prompt_tokens = model.to_tokens(prompt)
    prompt_len = prompt_tokens.shape[1]

    with torch.no_grad():
        output_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
        )

    if output_tokens.shape[1] <= prompt_len:
        return extract_last_token_residuals(model, prompt, layers)

    output_tokens = output_tokens.clone()
    _, cache = model.run_with_cache(output_tokens)

    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        act = cache[hook_name].squeeze(0)
        response_acts = act[prompt_len:]
        activations[layer] = response_acts.mean(dim=0)

    return activations


def extract_batch_response_mean(
    model: HookedTransformer,
    prompts: list[str],
    max_new_tokens: int = 50,
    layers: list[int] | None = None,
) -> list[dict[int, torch.Tensor]]:
    return [
        extract_response_mean_residuals(model, p, max_new_tokens, layers)
        for p in prompts
    ]
