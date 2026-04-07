import os
from pathlib import Path

import torch
from sparsify.sparse_coder import EncoderOutput


def save_activations(
    output_dir: Path,
    activations: dict[str, list[torch.Tensor]],
    start: int,
    end: int | float,
):

    torch.serialization.add_safe_globals([EncoderOutput])

    os.makedirs(output_dir, exist_ok=True)

    for layer, layer_activations in activations.items():
        file_path = output_dir / f"{layer}.{start}-{end}.pt"
        torch.save(layer_activations, file_path)


def save_sae_features(output_dir: Path, layer: str, sae_features: list[EncoderOutput]):
    torch.serialization.add_safe_globals([EncoderOutput])

    os.makedirs(output_dir, exist_ok=True)

    file_path = output_dir / f"{layer}.pt"
    torch.save(sae_features, file_path)
