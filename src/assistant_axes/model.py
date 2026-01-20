import torch
from transformer_lens import HookedTransformer


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str | None = None) -> HookedTransformer:
    device = device or get_device()
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device=device,
    )
    model.eval()
    return model
