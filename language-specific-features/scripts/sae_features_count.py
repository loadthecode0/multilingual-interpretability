import argparse
import math
import os
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from const import layer_to_index
from loader import load_activations
from tqdm.auto import tqdm
from utils import TqdmLoggingHandler, get_project_dir, set_deterministic


class Args(TypedDict):
    output_type: str
    hidden_dim: int | None
    layers: list[str]
    dataset_configs: list[str]
    in_dir: Path
    in_path: Path
    out_dir: Path
    out_path: Path
    shuffle_words: bool
    romanized: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Count sae features from a dataset for particular layers and languages."
    )

    parser.add_argument(
        "--output-type",
        help="Output type of the loaded file",
        type=str,
        choices=[
            "EncoderOutput",
        ],
    )

    parser.add_argument(
        "--hidden-dim",
        help="hidden dimension of the model (if omitted, inferred from the first activation file)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--dataset-configs",
        help='dataset configurations "dataset_name:config_name". Support bracex expansion',
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--in-dir",
        help="input directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--in-path",
        help="input path",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-path",
        help="output path",
        type=Path,
    )

    parser.add_argument(
        "--shuffle-words",
        help="Read inputs from shuffled-language directories (e.g., en_shuffled)",
        action="store_true",
    )

    parser.add_argument(
        "--romanized",
        help="Use romanized version of the data (Dakshina / icu_dakshina)",
        action="store_true",
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    processed_dataset_configs = []

    for dataset_config in args.dataset_configs:
        processed_dataset_configs.extend(expand(dataset_config))

    return {
        "output_type": args.output_type,
        "hidden_dim": args.hidden_dim,
        "dataset_configs": processed_dataset_configs,
        "layers": processed_layers,
        "in_dir": args.in_dir,
        "in_path": args.in_path,
        "out_dir": args.out_dir,
        "out_path": args.out_path,
        "shuffle_words": args.shuffle_words,
        "romanized": args.romanized,
    }


def count_sae_features(
    activations_list: list,
    layer_index: int,
    over_zero_token: torch.Tensor,
    over_zero_example: torch.Tensor,
    over_zero_total: torch.Tensor,
    num_examples: int,
    num_tokens: int,
    max_active_over_zero: torch.Tensor,
    min_active_over_zero: torch.Tensor,
    output_type: str,
    rounding_digit: int = 3,
):
    if output_type == "EncoderOutput":
        for activations in activations_list:
            top_acts = activations.top_acts
            top_indices = activations.top_indices

            num_examples += 1
            num_tokens += top_indices.shape[1]  # (batch, tokens, top_K)

            flat_top_indices = top_indices.flatten()
            flat_top_acts = top_acts.flatten()

            unique_feature_indices = torch.unique(flat_top_indices)
            over_zero_example[layer_index, unique_feature_indices] += 1

            ones = torch.ones_like(flat_top_indices, dtype=over_zero_token.dtype)
            over_zero_token[layer_index].scatter_add_(0, flat_top_indices, ones)

            over_zero_total[layer_index].scatter_add_(0, flat_top_indices, flat_top_acts)

            rounded_acts = flat_top_acts.round(decimals=rounding_digit)

            max_active_over_zero[layer_index].scatter_reduce_(
                0,
                flat_top_indices,
                rounded_acts,
                reduce="amax",
                include_self=True,
            )

            min_active_over_zero[layer_index].scatter_reduce_(
                0,
                flat_top_indices,
                rounded_acts,
                reduce="amin",
                include_self=True,
            )

    return (
        over_zero_token,
        over_zero_example,
        over_zero_total,
        num_examples,
        num_tokens,
        max_active_over_zero,
        min_active_over_zero,
    )


def _infer_hidden_dim(activations_list: list, output_type: str) -> int:
    if not activations_list:
        raise ValueError(
            "Cannot infer hidden dimension because no activations were loaded."
        )

    if output_type == "EncoderOutput":
        for activations in activations_list:
            top_indices = getattr(activations, "top_indices", None)
            if top_indices is not None:
                return top_indices.shape[-1]

    raise ValueError(
        "Unable to infer hidden dimension from activations. Please provide --hidden-dim explicitly."
    )


def main(args: Args):
    num_layers = len(args["layers"])
    provided_hidden_dim = args["hidden_dim"]

    for dataset_config in tqdm(args["dataset_configs"], desc="Processing languages"):
        dataset, lang = dataset_config.split(":")

        over_zero_token = None
        over_zero_example = None
        over_zero_total = None
        max_active_over_zero = None
        min_active_over_zero = None
        hidden_dim = provided_hidden_dim

        # Build language suffix (e.g., "en", "en_romanized", or "en_shuffled")
        lang_suffix_parts = [lang]
        if args["romanized"]:
            lang_suffix_parts.append("romanized")
        if args["shuffle_words"]:
            lang_suffix_parts.append("shuffled")
        lang_suffix = "_".join(lang_suffix_parts)

        for layer in tqdm(args["layers"], desc="Processing layers", leave=False):
            num_examples = 0
            num_tokens = 0

            input_dir = args["in_dir"] / args["in_path"] / dataset / lang_suffix
            activations = load_activations(input_dir, layer, logger)

            if over_zero_token is None:
                hidden_dim = hidden_dim or _infer_hidden_dim(
                    activations, args["output_type"]
                )
                over_zero_token = torch.zeros((num_layers, hidden_dim))
                over_zero_example = torch.zeros((num_layers, hidden_dim))
                over_zero_total = torch.zeros((num_layers, hidden_dim))
                max_active_over_zero = torch.zeros((num_layers, hidden_dim))
                min_active_over_zero = torch.full((num_layers, hidden_dim), math.inf)

            layer_index = layer_to_index[layer]
            (
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
            ) = count_sae_features(
                activations,
                layer_index,
                over_zero_token,
                over_zero_example,
                over_zero_total,
                num_examples,
                num_tokens,
                max_active_over_zero,
                min_active_over_zero,
                args["output_type"],
            )

        output_dir = args["out_dir"] / args["out_path"] / dataset
        filename_parts = [lang]
        if args["romanized"]:
            filename_parts.append("romanized")
        if args["shuffle_words"]:
            filename_parts.append("shuffled")
        filename = "_".join(filename_parts) + ".pt"
        file_path = output_dir / filename

        os.makedirs(output_dir, exist_ok=True)

        # Save as a sparse tensor
        min_active_over_zero[min_active_over_zero == math.inf] = 0

        output = {
            "num_examples": num_examples,
            "num_tokens": num_tokens,
            "over_zero_token": over_zero_token.to_sparse(),
            "over_zero_example": over_zero_example.to_sparse(),
            "over_zero_total": over_zero_total.to_sparse(),
            "max_active_over_zero": max_active_over_zero.to_sparse(),
            "min_active_over_zero": min_active_over_zero.to_sparse(),
        }

        torch.save(output, file_path)


if __name__ == "__main__":
    set_deterministic()

    logger = TqdmLoggingHandler.get_logger("activations_count")

    args = parse_args()
    main(args)
