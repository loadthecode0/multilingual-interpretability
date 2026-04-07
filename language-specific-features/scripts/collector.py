from pathlib import Path
from typing import TypedDict

import torch
from datasets import Dataset
from loader import load_sae
from nnsight import LanguageModel
from sparsify import Sae
from sparsify.sparse_coder import EncoderOutput
from tqdm.auto import tqdm
from utils import get_device, get_nested_attr


def _get_topk_from_encoding(sae_encoding):
    if isinstance(sae_encoding, torch.Tensor):  # Gemma returns raw activations
        total_features = sae_encoding.shape[-1]
        k = min(200, total_features)
        return torch.topk(sae_encoding, k=k, dim=-1)

    has_top_acts = hasattr(sae_encoding, "top_acts")
    has_top_indices = hasattr(sae_encoding, "top_indices")

    if has_top_acts and has_top_indices:  # Llama-style encoders
        return sae_encoding.top_acts, sae_encoding.top_indices

    # feature_acts = getattr(sae_encoding, "feature_acts", None)
    # feature_acts = sae_encoding
    # if feature_acts is None:
    #     raise AttributeError(
    #         f"SAE encoding output must provide either (top_acts, top_indices) "
    #         f"or feature_acts to derive them. Available attributes: {sae_encoding.__dict__.keys()}"
    #     )

    # top_k = min(20, feature_acts.shape[-1])
    # return torch.topk(feature_acts, k=top_k, dim=-1)


def collect_activations(
    llm: LanguageModel, layers: list[str], prompt: str
) -> dict[str, list[torch.Tensor]]:

    # print(llm)
    layers_modules = {layer: get_nested_attr(llm, layer) for layer in layers}

    layers_activations = {layer: None for layer in layers}

    with llm.trace(prompt):
        for layer in layers:
            layers_activations[layer] = layers_modules[layer].output.cpu().save()

    layers_activations_processed = {
        layer: (
            activations[0].value
            if isinstance(activations, tuple)
            else activations.value
        )
        for layer, activations in layers_activations.items()
    }

    return layers_activations_processed


def sae_features_from_activations(
    activations_list: list[torch.Tensor],
    sae: Sae,
    device: torch.device,
    batch: int = 100,
):
    activations_size = [
        activations.shape[1] for activations in activations_list
    ]  # [a, b, ...]
    activations_list = torch.cat(activations_list, dim=1)  # tensor(1, a+b+..., 2048)

    top_acts = []
    top_indices = []
    # list[tensor(1, batch, 2048), ...]
    chunks = torch.split(activations_list, batch, dim=1)

    for chunk in chunks:
        sae_features = sae.encode(chunk.squeeze(0).to(device))
        if isinstance(sae_features, torch.Tensor):
            nonzero_count = int(torch.count_nonzero(sae_features).item())
            total_values = sae_features.numel()
            has_nonzero = nonzero_count > 0
            print(
                f"[DEBUG] SAE encode chunk -> nonzero activations: "
                f"{nonzero_count}/{total_values} (any_nonzero={has_nonzero})"
            )
        print(sae_features)
        chunk_top_acts, chunk_top_indices = _get_topk_from_encoding(sae_features)
        top_acts.append(chunk_top_acts.unsqueeze(0).cpu())
        top_indices.append(chunk_top_indices.unsqueeze(0).cpu())

    top_acts = torch.cat(top_acts, dim=1)
    top_acts = torch.split(top_acts, activations_size, dim=1)

    top_indices = torch.cat(top_indices, dim=1)
    top_indices = torch.split(top_indices, activations_size, dim=1)

    all_sae_features = []

    for top_acts, top_indices in zip(top_acts, top_indices):
        all_sae_features.append(EncoderOutput(top_acts, top_indices, None))

    return all_sae_features


class DatasetConfig(TypedDict):
    name: str
    lang: str
    split: str
    start: int
    end: int | float


def collect_all_activations(
    llm: LanguageModel,
    layers: list[str],
    dataset: Dataset,
    prompt_template: str,
    shuffle_words: bool = False,
    romanized: bool = False,
    dataset_name: str = "",
) -> dict[str, list[torch.Tensor]]:
    all_activations = {layer: [] for layer in layers}
    example_count = 0

    for row in tqdm(dataset, desc="Processing Samples", leave=False):
        # Choose prompt based on romanized flag
        if romanized and dataset_name in {"dakshina", "icu_dakshina"}:
            prompt = row["romanized"]  # Use romanized field directly
        else:
            prompt = prompt_template.format_map(row)

        if example_count < 3:
            tqdm.write(f"Prompt: {prompt}")
            example_count += 1
        
        # Shuffle words if requested
        if shuffle_words:
            import random
            original_prompt = prompt
            words = prompt.split()
            random.shuffle(words)
            prompt = " ".join(words)
            # Print first few examples to show shuffling
            if example_count < 3:
                print(f"[SHUFFLE DEBUG] Example {example_count + 1}:")
                print(f"  Original: {original_prompt}")
                print(f"  Shuffled: {prompt}")
                print()
            
        activations = collect_activations(llm, layers, prompt)

        for layer, layer_activations in activations.items():
            all_activations[layer].append(layer_activations)
        
        example_count += 1

    return all_activations


def collect_all_sae_features(
    all_activations: dict[str, list[torch.Tensor]],
    layers: list[str],
    model: str,
    sae_model: str,
    local_sae_dir: Path,
    batch: int,
):
    all_sae_features = {layer: [] for layer in layers}

    device = get_device()

    for layer, layer_activations in tqdm(all_activations.items()):
        sae = load_sae(model, sae_model, layer, local_sae_dir).to(device)
        all_sae_features[layer] = sae_features_from_activations(
            layer_activations, sae, device, batch
        )

    return all_sae_features
