"""
Utilities for loading Gemma-Scope sparse autoencoders.

This mirrors the logic used in the probing scripts so that we can
reuse the same SCoPE weights inside the causal intervention pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


class JumpReluSae(nn.Module):
    """Minimal Jump-ReLU SAE implementation from the GemmaScope tutorial."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model
        self.d_sae = d_sae

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(acts))

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        sae_subdir: str,
        device: str | torch.device,
    ) -> "JumpReluSae":
        params_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{sae_subdir}/params.npz",
            force_download=False,
        )
        params = np.load(params_path)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        model.to(device)
        return model


def normalize_repo_id(repo_id: str) -> str:
    if repo_id.startswith("google/"):
        return repo_id
    return f"google/{repo_id}"


def load_gemma_scope_autoencoder(
    repo_id: str,
    layer_idx: int,
    width: str = "65k",
    average_l0: int = 52,
    device: str | torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> JumpReluSae:
    """
    Load a single Gemma-Scope SAE for the requested layer.
    """

    scoped_repo = normalize_repo_id(repo_id)
    sae_subdir = f"layer_{layer_idx}/width_{width}/average_l0_{average_l0}"

    sae = JumpReluSae.from_pretrained(scoped_repo, sae_subdir, device)
    sae.to(dtype)
    sae.eval()
    return sae












