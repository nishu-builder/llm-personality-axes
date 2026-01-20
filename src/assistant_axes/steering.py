from typing import Callable

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def make_steering_hook(
    direction: torch.Tensor,
    scale: float,
    position: int | None = None,
) -> Callable:
    direction = direction / direction.norm()

    def hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        if position is None:
            activation = activation + scale * direction
        else:
            activation[:, position, :] = activation[:, position, :] + scale * direction
        return activation

    return hook


def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    direction: torch.Tensor,
    layer: int,
    scale: float,
    max_new_tokens: int = 100,
) -> str:
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = make_steering_hook(direction.to(model.cfg.device), scale)

    tokens = model.to_tokens(prompt)

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
        )

    output_text = model.to_string(output_tokens[0])
    return output_text


def generate_baseline(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    tokens = model.to_tokens(prompt)
    output_tokens = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        verbose=False,
    )
    return model.to_string(output_tokens[0])
