from pathlib import Path
from typing import Callable

import torch
from transformer_lens import HookedTransformer

from assistant_axes.model import MODELS, load_model


MODEL_DEFAULTS = {
    "qwen": {
        "directions": "artifacts/directions/qwen_last_token.pt",
        "layers": range(5, 30),
        "threshold": 3.0,
    },
    "llama": {
        "directions": "artifacts/directions/llama_last_token.pt",
        "layers": range(9, 20),
        "threshold": 2.0,
    },
}


class CappedModel:
    """
    Wraps a HookedTransformer and applies activation capping during generation.

    Capping keeps the model's activations from drifting too far from a target
    personality (e.g., "assistant"). Only intervenes when activations fall
    below the threshold - leaves normal behavior untouched.

    Example:
        model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        capped = CappedModel(model, directions, layers=range(12, 24), threshold=13.0)
        output = capped.generate("User: Tell me a joke\\n\\nAssistant:")
    """

    def __init__(
        self,
        model: HookedTransformer,
        directions: dict[int, torch.Tensor],
        layers: list[int] | range | None = None,
        threshold: float = 13.0,
    ):
        self.model = model
        self.threshold = threshold

        if layers is None:
            layers = list(directions.keys())
        self.layers = list(layers)

        self.directions = {}
        for layer in self.layers:
            if layer not in directions:
                raise ValueError(f"No direction provided for layer {layer}")
            d = directions[layer]
            d = d / d.norm()
            self.directions[layer] = d.to(model.cfg.device)

    def _make_capping_hook(self, direction: torch.Tensor) -> Callable:
        threshold = self.threshold

        def hook(activation: torch.Tensor, hook) -> torch.Tensor:
            d = direction.to(dtype=activation.dtype, device=activation.device)
            proj = (activation @ d).unsqueeze(-1)
            below = proj < threshold
            correction = (threshold - proj) * d
            return activation + below.to(activation.dtype) * correction

        return hook

    def _get_hooks(self) -> list[tuple[str, Callable]]:
        hooks = []
        for layer in self.layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = self._make_capping_hook(self.directions[layer])
            hooks.append((hook_name, hook_fn))
        return hooks

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        tokens = self.model.to_tokens(prompt)

        with self.model.hooks(fwd_hooks=self._get_hooks()):
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", False),
                verbose=kwargs.get("verbose", False),
            )

        return self.model.to_string(output_tokens[0])

    def generate_uncapped(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """Generate without capping, for comparison."""
        tokens = self.model.to_tokens(prompt)
        output_tokens = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=kwargs.get("do_sample", False),
            verbose=kwargs.get("verbose", False),
        )
        return self.model.to_string(output_tokens[0])

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        directions_path: str | Path,
        layers: list[int] | range | None = None,
        threshold: float = 13.0,
        **model_kwargs,
    ) -> "CappedModel":
        """
        Load a model and directions in one call.

        Example:
            capped = CappedModel.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                "data/directions/assistant_directions.pt",
                layers=range(12, 24),
            )
        """
        device = model_kwargs.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = HookedTransformer.from_pretrained(
            model_name,
            dtype=model_kwargs.get("dtype", torch.bfloat16),
            device=device,
        )
        model.eval()

        directions = torch.load(directions_path, weights_only=True)

        return cls(model, directions, layers=layers, threshold=threshold)

    @classmethod
    def from_model_key(
        cls,
        model_key: str = "qwen",
        **overrides,
    ) -> "CappedModel":
        """
        Load a capped model with sensible defaults.

        Uses pre-computed directions and optimal layer ranges for each model.

        Args:
            model_key: "qwen" or "llama"
            **overrides: Override any default (directions, layers, threshold)

        Example:
            capped = CappedModel.from_model_key("qwen")
            capped = CappedModel.from_model_key("llama", threshold=3.0)
        """
        if model_key not in MODEL_DEFAULTS:
            raise ValueError(f"Unknown model_key: {model_key}. Use one of {list(MODEL_DEFAULTS.keys())}")

        defaults = MODEL_DEFAULTS[model_key]
        directions_path = overrides.get("directions", defaults["directions"])
        layers = overrides.get("layers", defaults["layers"])
        threshold = overrides.get("threshold", defaults["threshold"])

        model = load_model(model_key)
        directions = torch.load(directions_path, weights_only=True)

        return cls(model, directions, layers=layers, threshold=threshold)
