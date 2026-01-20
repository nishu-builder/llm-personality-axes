import torch
from transformer_lens import HookedTransformer


MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}

DEFAULT_MODEL = "qwen"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_key: str = DEFAULT_MODEL,
    device: str | None = None,
) -> HookedTransformer:
    device = device or get_device()
    model_name = MODELS.get(model_key, model_key)
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device=device,
    )
    model.eval()
    return model
