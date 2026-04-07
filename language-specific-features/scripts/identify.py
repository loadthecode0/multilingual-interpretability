import argparse
import bisect
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from const import (
    lang_choices_to_qualified_name,
    layer_to_index,
    model_choices,
    sae_model_choices,
)
from loader import load_sae
from tqdm.auto import tqdm
from utils import TqdmLoggingHandler, get_project_dir, set_deterministic


class Args(TypedDict):
    model: str
    sae_model: str
    local_sae_dir: Path | None
    layers: list[str]
    dataset_configs: list[str]
    in_dir: Path
    in_path: Path
    out_dir: Path
    out_path: Path
    out_filename: Path
    top: int
    top_per_layer: bool
    topk_threshold_ratio: float
    top_by_frequency: bool
    example_rate: float
    entropy_threshold: float
    lang_specific: bool
    algorithm: str
    shuffle_words: bool
    romanized: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Identify language-specific features or language-specific neurons in a model."
    )

    parser.add_argument(
        "--model",
        help="model name",
        type=str,
        default=model_choices[0],
        choices=model_choices,
    )

    parser.add_argument(
        "--sae-model",
        help="sae model name",
        type=str,
        default=None,
        choices=sae_model_choices,
    )

    parser.add_argument(
        "--local-sae-dir",
        help="local directory to load SAE model from",
        type=Path,
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
        "--out-filename",
        help="output path",
        type=Path,
        default="lape.pt",
    )

    parser.add_argument(
        "--top",
        help="Top N features to select for each language sorted by entropy. 0 means all features.",
        type=int,
    )

    parser.add_argument(
        "--top-per-layer",
        help="Top N features to select for each layer sorted by entropy.",
        action="store_true",
    )

    parser.add_argument(
        "--topk-threshold-ratio",
        help="Threshold ratio for topk features relative to the max value.",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-by-frequency",
        help="Whether to find language-specific features and sorted by frequency.",
        action="store_true",
    )

    parser.add_argument(
        "--example-rate",
        help="Minimum number of examples for a feature to be considered as a language-specific feature.",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--entropy-threshold",
        help="Minimum/Maximum entropy threshold for a feature to be considered as a language-specific/language-agnostic feature.",
        type=float,
    )

    parser.add_argument(
        "--lang-specific",
        help="Whether to find language-specific features or not.",
        action="store_true",
    )

    parser.add_argument(
        "--lang-shared",
        help="Whether to find language-specific features or not.",
        action="store_true",
    )

    parser.add_argument(
        "--shared-count",
        help="Count of shared features across languages.",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--algorithm",
        help="Algorithm to use for the analysis.",
        type=str,
        default="sae_lape",
        choices=["sae_lape", "lape"],
    )

    parser.add_argument(
        "--shuffle-words",
        help="Shuffle words inside each sentence before tokenization",
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
        "model": args.model,
        "sae_model": args.sae_model,
        "local_sae_dir": args.local_sae_dir,
        "layers": processed_layers,
        "dataset_configs": processed_dataset_configs,
        "in_dir": args.in_dir,
        "in_path": args.in_path,
        "out_dir": args.out_dir,
        "out_path": args.out_path,
        "out_filename": args.out_filename,
        "top": args.top,
        "top_per_layer": args.top_per_layer,
        "topk_threshold_ratio": args.topk_threshold_ratio,
        "top_by_frequency": args.top_by_frequency,
        "example_rate": args.example_rate,
        "entropy_threshold": args.entropy_threshold,
        "lang_specific": args.lang_specific,
        "lang_shared": args.lang_shared,
        "shared_count": args.shared_count,
        "algorithm": args.algorithm,
        "shuffle_words": args.shuffle_words,
        "romanized": args.romanized,
    }


def aggregate_activations_count(
    lang_to_activations_count, normalized_lang, activations_count
):
    prev_num_examples = lang_to_activations_count[normalized_lang].get(
        "num_examples", 0
    )
    prev_num_tokens = lang_to_activations_count[normalized_lang].get("num_tokens", 0)
    prev_over_zero_token = lang_to_activations_count[normalized_lang].get(
        "over_zero_token", 0
    )
    prev_over_zero_example = lang_to_activations_count[normalized_lang].get(
        "over_zero_example", 0
    )
    prev_over_zero_total = lang_to_activations_count[normalized_lang].get(
        "over_zero_total", 0
    )

    prev_max_active_over_zero = lang_to_activations_count[normalized_lang].get(
        "max_active_over_zero",
        activations_count["max_active_over_zero"].to_dense(),
    )

    current_min_active_over_zero = activations_count["min_active_over_zero"].to_dense()
    current_min_active_over_zero[current_min_active_over_zero == 0] = math.inf
    prev_min_active_over_zero = lang_to_activations_count[normalized_lang].get(
        "min_active_over_zero",
        current_min_active_over_zero,
    )

    lang_to_activations_count[normalized_lang] = {
        "num_examples": prev_num_examples + activations_count["num_examples"],
        "num_tokens": prev_num_tokens + activations_count["num_tokens"],
        "over_zero_token": prev_over_zero_token
        + activations_count["over_zero_token"].to_dense(),
        "over_zero_example": prev_over_zero_example
        + activations_count["over_zero_example"].to_dense(),
        "over_zero_total": prev_over_zero_total
        + activations_count["over_zero_total"].to_dense(),
        "max_active_over_zero": torch.max(
            prev_max_active_over_zero,
            activations_count["max_active_over_zero"].to_dense(),
        ),
        "min_active_over_zero": torch.min(
            prev_min_active_over_zero,
            current_min_active_over_zero,
        ),
    }


def stack_activations_count(
    lang_to_activations_count, sorted_lang
) -> tuple[list, list, torch.Tensor]:
    """Stack activations count for each language.

    Returns:
        tuple[list, list, torch.Tensor]: num_examples, num_tokens, over_zero_token
            num_examples: number of examples for each language
            num_tokens: number of tokens for each language
            over_zero_token: number of activations over zero for each language with shape (layer x hidden_dim x lang)
    """
    num_examples = []
    num_tokens = []
    over_zero_token = []
    over_zero_example = []
    global_max_active_over_zero = None
    global_min_active_over_zero = None

    global_over_zero_total = 0
    global_over_zero_token = 0

    for lang in sorted_lang:
        num_examples.append(lang_to_activations_count[lang]["num_examples"])
        num_tokens.append(lang_to_activations_count[lang]["num_tokens"])
        over_zero_token.append(lang_to_activations_count[lang]["over_zero_token"])
        over_zero_example.append(lang_to_activations_count[lang]["over_zero_example"])

        global_over_zero_total += lang_to_activations_count[lang]["over_zero_total"]
        global_over_zero_token += lang_to_activations_count[lang]["over_zero_token"]

        if global_max_active_over_zero is None:
            global_max_active_over_zero = lang_to_activations_count[lang][
                "max_active_over_zero"
            ]
        else:
            global_max_active_over_zero = torch.max(
                global_max_active_over_zero,
                lang_to_activations_count[lang]["max_active_over_zero"],
            )

        if global_min_active_over_zero is None:
            global_min_active_over_zero = lang_to_activations_count[lang][
                "min_active_over_zero"
            ]
        else:
            global_min_active_over_zero = torch.min(
                global_min_active_over_zero,
                lang_to_activations_count[lang]["min_active_over_zero"],
            )

    num_examples = torch.tensor(num_examples)
    num_tokens = torch.tensor(num_tokens)
    over_zero_token = torch.stack(over_zero_token, dim=-1)
    over_zero_example = torch.stack(over_zero_example, dim=-1)

    global_avg_active_over_zero = global_over_zero_total / global_over_zero_token

    return (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    )


def sae_lape(
    num_examples,
    num_tokens,
    over_zero_token,
    over_zero_example,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    topk_threshold_ratio,
    example_rate,
    top,
    top_per_layer,
    sorted_lang,
    entropy_threshold,
    lang_specific,
    lang_shared,
    shared_count,
    top_by_frequency,
):
    num_layers, hidden_dim, lang = over_zero_token.size()

    logger.info(f"num_layers: {num_layers}, hidden_dim: {hidden_dim}, lang: {lang}")

    # over_zero_token (layer x hidden_dim x lang) is divided by num_tokens (lang) to get activation probability
    activation_probs = over_zero_token / num_tokens

    # L1 normalization
    normed_activation_probs = activation_probs / activation_probs.sum(
        dim=-1, keepdim=True
    )
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    # entropy calculation
    log_probs = torch.where(
        normed_activation_probs > 0, normed_activation_probs.log(), 0
    )
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)

    # Take the N smallest or largest values of the entropy tensor
    # If largest is True, take the N largest values, otherwise take the N smallest values.
    largest = False

    # If there are NaN values in the entropy tensor, raise an error
    if torch.isnan(entropy).sum():
        logger.error(f"NaN values in entropy tensor: {torch.isnan(entropy).sum()}")
        raise ValueError

    # Dismiss the neruon if no language has a total activated examples over the threshold
    num_examples = num_examples.clone().detach().unsqueeze(0) * example_rate
    num_examples = num_examples.type(torch.int)
    over_zero_example_dismissed_neurons = (over_zero_example >= num_examples).any(
        dim=-1
    )

    logger.info(f"num_examples after thresholding: {num_examples.tolist()}")

    # Dismiss the neuron if no language has a total activated tokens over the threshold
    # total activated tokens > 10% of the total tokens for HFLs
    hfl_rate = 0.1
    num_tokens = num_tokens.clone().detach().unsqueeze(0) * hfl_rate
    num_tokens = num_tokens.type(torch.int)
    over_zero_token_dismissed_neurons = (over_zero_token > num_tokens).any(dim=-1)

    logger.info(f"num_tokens after thresholding: {num_tokens.tolist()}")

    # Set the entropy of the dismissed neuron
    dismissed_neurons = (
        over_zero_example_dismissed_neurons * over_zero_token_dismissed_neurons
    )
    patched_val = -torch.inf if largest else torch.inf
    entropy[dismissed_neurons == 0] = patched_val

    min_entropy = entropy[entropy != patched_val].min() if largest else entropy.min()
    max_entropy = (
        entropy[entropy != patched_val].max()
        if not largest
        else (entropy != patched_val).min()
    )
    non_inf_entropy = (entropy != patched_val).sum().item()

    logger.info(
        f"Num of non inf entropy: {non_inf_entropy} - Min entropy: {min_entropy} - Max entropy: {max_entropy}"
    )

    # Take the N smallest or largest values of the entropy tensor for each language
    flattened_entropy = entropy.flatten()
    non_inf_entropy = (entropy != patched_val).sum().item()

    # Filter non inf entropy
    values, indices = flattened_entropy.topk(non_inf_entropy, largest=largest)

    logger.info(f"Top entropy value: {values.round(decimals=5)[:non_inf_entropy]}")

    # Convert the index to the row and column index of the entropy tensor
    layer_index = indices // entropy.size(1)
    hidden_dim_index = indices % entropy.size(1)
    selected_probs = activation_probs[layer_index, hidden_dim_index]  # topk x lang

    logger.info(f"Selected probs size: {selected_probs.size()}")
    logger.info(
        f"Selected probs bincount per lang index: {torch.bincount(selected_probs.argmax(dim=-1))}"
    )
    merged_index = torch.stack((layer_index, hidden_dim_index), dim=-1)

    # selected_probs shape = (lang x topk)
    selected_probs = selected_probs.transpose(0, 1)

    # DEBUG: Save selected_probs as a CSV file
    # csv_output_path = args["out_dir"] / args["out_path"] / "selected_probs.csv"
    # os.makedirs(csv_output_path.parent, exist_ok=True)

    # df = pd.DataFrame(selected_probs.numpy())
    # df.to_csv(csv_output_path, index=False)

    # logger.info(f"Selected probabilities saved to {csv_output_path}")

    # DEBUG: Check the selected probabilities for a specific target
    # target = torch.tensor([11, 8452])
    # index = (merged_index == target).all(dim=1).nonzero(as_tuple=True)[0].item()

    # logger.info(f"selected probs for {target}: {selected_probs[:, index].round(decimals=2)}")

    # DEBUG: check the 95th percentile of the selected probabilities
    # flattened_probs = activation_probs.flatten()
    # activation_bar_ratio = 0.95
    # flattened_probs
    # activation_bar = flattened_probs.kthvalue(
    #     round(len(flattened_probs) * activation_bar_ratio)
    # ).values.item()

    # logger.info(f"95th percentile activation prob: {activation_bar}")

    # selected_probs_95th = selected_probs >= activation_bar

    # logger.info(
    #     f"lang-features count: {Counter(selected_probs_95th.sum(dim=0).tolist())}"
    # )

    # Create mask for elements >= N% of max in their column
    # True means features are considered as specific for that language
    max_selected_probs = selected_probs.max(dim=0, keepdim=True).values
    mask_selected_probs = selected_probs >= (max_selected_probs * topk_threshold_ratio)

    # Check shared features across languages
    count_shared_features = mask_selected_probs.sum(dim=0)
    count_shared_indices = torch.nonzero(count_shared_features > 1).squeeze()
    
    # Ensure it's at least 1-dimensional for indexing
    if count_shared_indices.numel() > 0 and count_shared_indices.dim() == 0:
        count_shared_indices = count_shared_indices.unsqueeze(0)

    if lang_specific:
        mask_selected_probs[:, count_shared_indices] = False

        # Check entropies for each count of shared features
        for count in range(1, len(sorted_lang) + 1):
            count_shared_features_indices = torch.nonzero(
                count_shared_features == count
            ).squeeze()

            if count_shared_features_indices.numel() == 0:
                continue

            # Ensure it's at least 1-dimensional for indexing
            if count_shared_features_indices.dim() == 0:
                count_shared_features_indices = count_shared_features_indices.unsqueeze(0)

            count_shared_features_indices = [
                tuple(row.tolist())
                for row in merged_index[count_shared_features_indices]
            ]

            row_idx, col_idx = zip(*count_shared_features_indices)
            count_entropies = entropy[row_idx, col_idx].flatten()
            sorted_count_entropies = count_entropies.sort().values.round(decimals=3)

            logger.info(f"Shared features entropy {count}\n{sorted_count_entropies}")
    elif lang_shared:
        non_specific_count_indices = torch.nonzero(
            count_shared_features != shared_count
        ).squeeze()
        
        # Ensure it's at least 1-dimensional for indexing
        if non_specific_count_indices.numel() > 0 and non_specific_count_indices.dim() == 0:
            non_specific_count_indices = non_specific_count_indices.unsqueeze(0)
        
        mask_selected_probs[:, non_specific_count_indices] = False

        logger.info(f"Shared features: {shared_count}")

    logger.info(
        f"features count: {count_shared_features.size(0)}\n\t{Counter(count_shared_features.tolist())}"
    )

    # Get the indices of the selected features for each language
    lang, topk_indice = torch.where(mask_selected_probs)

    lang_bincount = torch.bincount(lang).tolist()
    lang_topk_indice = topk_indice.split(lang_bincount)

    # Store the indices of the selected neurons for each language
    final_indice = []
    final_indice_global_max_active = []
    final_indice_global_min_active = []
    final_indice_global_avg_active = []
    features_info = {}

    for i, indices in enumerate(lang_topk_indice):
        lang_indices = [tuple(row.tolist()) for row in merged_index[indices]]

        if top:
            if top_by_frequency:
                # Get the indices of the selected features for each language
                matches = (
                    merged_index[:, None, :] == torch.tensor(lang_indices)[None, :, :]
                ).all(dim=2)
                matching_rows = matches.any(dim=1)
                selected_indices = torch.nonzero(matching_rows).squeeze()
                
                # Ensure it's at least 1-dimensional for indexing
                if selected_indices.dim() == 0:
                    selected_indices = selected_indices.unsqueeze(0)

                lang_probs = selected_probs[i, selected_indices]
                lang_indices = sorted(
                    lang_indices,
                    key=lambda indices: lang_probs[lang_indices.index(indices)],
                    reverse=True,
                )

            if top_per_layer:
                count_layer = [0] * num_layers
                new_lang_indices = []

                for lang_index in lang_indices:
                    layer_index = lang_index[0]

                    if count_layer[layer_index] < top:
                        new_lang_indices.append(lang_index)
                        count_layer[layer_index] += 1

                lang_indices = new_lang_indices
            else:
                lang_indices = lang_indices[:top]

        if entropy_threshold:
            row_idx, col_idx = zip(*lang_indices)
            lang_index_entropy = entropy[row_idx, col_idx].tolist()
            lang_index_entropy_threshold_index = bisect.bisect_right(
                lang_index_entropy, entropy_threshold
            )
            lang_indices = lang_indices[:lang_index_entropy_threshold_index]

        lang_indices.sort()

        if len(lang_indices) == 0:
            print(f"No features selected for language {sorted_lang[i]}")
            continue

        row_idx, col_idx = zip(*lang_indices)
        entropy_values = entropy[row_idx, col_idx]
        min_entropy = entropy_values.min().item()
        max_entropy = entropy_values.max().item()

        logger.info(
            f"Language {sorted_lang[i]}\n\tFeatures selected: {len(lang_indices)}\n\tTopk features: {lang_indices}\nmin_entropy: {min_entropy}\nmax_entropy: {max_entropy}"
        )

        logger.info(f"Features selected entropies {entropy_values.round(decimals=2)}")

        # Get the indices of the selected features for each language
        matches = (
            merged_index[:, None, :] == torch.tensor(lang_indices)[None, :, :]
        ).all(dim=2)
        matching_rows = matches.any(dim=1)
        selected_indices = torch.nonzero(matching_rows).squeeze()
        
        # Ensure it's at least 1-dimensional for indexing
        if selected_indices.dim() == 0:
            selected_indices = selected_indices.unsqueeze(0)

        logger.info(
            f"Features selected probs {selected_probs[i, selected_indices].round(decimals=2)}"
        )

        features_info[sorted_lang[i]] = {
            "indicies": lang_indices,
            "selected_probs": selected_probs[i, selected_indices],
            "entropies": entropy_values,
        }

        layer_index = [[] for _ in range(num_layers)]
        layer_index_max_active = [[] for _ in range(num_layers)]
        layer_index_min_active = [[] for _ in range(num_layers)]
        layer_index_avg_active = [[] for _ in range(num_layers)]

        for l, h in lang_indices:
            layer_index[l].append(h)
            layer_index_max_active[l].append(global_max_active_over_zero[l, h].item())
            layer_index_min_active[l].append(global_min_active_over_zero[l, h].item())
            layer_index_avg_active[l].append(global_avg_active_over_zero[l, h].item())

        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()

        for l, h in enumerate(layer_index_max_active):
            layer_index_max_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_min_active):
            layer_index_min_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_avg_active):
            layer_index_avg_active[l] = torch.tensor(h)

        final_indice.append(layer_index)
        final_indice_global_max_active.append(layer_index_max_active)
        final_indice_global_min_active.append(layer_index_min_active)
        final_indice_global_avg_active.append(layer_index_avg_active)

    return (
        final_indice,
        final_indice_global_max_active,
        final_indice_global_min_active,
        final_indice_global_avg_active,
        features_info,
    )


def lape(
    over_zero_token,
    num_tokens,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    sorted_lang,
):
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95

    num_layers, intermediate_size, lang_num = over_zero_token.size()

    activation_probs = over_zero_token / num_tokens  # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(
        dim=-1, keepdim=True
    )
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(
        normed_activation_probs > 0, normed_activation_probs.log(), 0
    )
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError

    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(
        round(len(flattened_probs) * filter_rate)
    ).values.item()

    print(top_prob_value)

    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # n x lang

    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))

    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(
        round(len(flattened_probs) * activation_bar_ratio)
    ).values.item()

    print((selected_probs > activation_bar).sum(dim=1).tolist())

    lang, indice = torch.where(selected_probs > activation_bar)

    final_indice = []
    final_indice_global_max_active = []
    final_indice_global_min_active = []
    final_indice_global_avg_active = []
    features_info = {}

    merged_index = torch.stack((row_index, col_index), dim=-1)

    for i, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]

        lang_index.sort()

        # Compare each row in tensor to each target
        # tensor[:, None, :] has shape [N, 1, D]
        # targets[None, :, :] has shape [1, M, D]
        # Broadcasting to [N, M, D]
        matches = (
            merged_index[:, None, :] == torch.tensor(lang_index)[None, :, :]
        ).all(dim=2)

        # Reduce across target rows (axis=1) to see if any matched
        matching_rows = matches.any(dim=1)

        # Get indices of matching rows
        merge_indices = torch.nonzero(matching_rows).squeeze()
        
        # Ensure it's at least 1-dimensional for indexing
        if merge_indices.dim() == 0:
            merge_indices = merge_indices.unsqueeze(0)

        row_idx, col_idx = zip(*lang_index)

        features_info[sorted_lang[i]] = {
            "indicies": lang_index,
            "selected_probs": selected_probs[i, merge_indices],
            "entropies": entropy[row_idx, col_idx],
        }

        layer_index = [[] for _ in range(num_layers)]
        layer_index_max_active = [[] for _ in range(num_layers)]
        layer_index_min_active = [[] for _ in range(num_layers)]
        layer_index_avg_active = [[] for _ in range(num_layers)]

        for l, h in lang_index:
            layer_index[l].append(h)
            layer_index_max_active[l].append(global_max_active_over_zero[l, h].item())
            layer_index_min_active[l].append(global_min_active_over_zero[l, h].item())
            layer_index_avg_active[l].append(global_avg_active_over_zero[l, h].item())

        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()

        for l, h in enumerate(layer_index_max_active):
            layer_index_max_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_min_active):
            layer_index_min_active[l] = torch.tensor(h)

        for l, h in enumerate(layer_index_avg_active):
            layer_index_avg_active[l] = torch.tensor(h)

        final_indice.append(layer_index)
        final_indice_global_max_active.append(layer_index_max_active)
        final_indice_global_min_active.append(layer_index_min_active)
        final_indice_global_avg_active.append(layer_index_avg_active)

    return (
        final_indice,
        final_indice_global_max_active,
        final_indice_global_min_active,
        final_indice_global_avg_active,
        features_info,
    )


def get_sae_features(
    final_indices, model_name, sae_model_name, layers, local_sae_dir: Path | None
):
    num_langs = len(final_indices)

    sae_features = [[] for _ in range(num_langs)]

    for layer in tqdm(layers):
        sae = load_sae(model_name, sae_model_name, layer, local_sae_dir)

        for lang in tqdm(range(num_langs)):
            layer_index = layer_to_index[layer]
            feature_indices = final_indices[lang][layer_index]
            features = sae.W_dec[feature_indices].detach()
            sae_features[lang].append(features)

    return sae_features


def main(args: Args):
    lang_to_activations_count = defaultdict(dict)

    for dataset_config in tqdm(args["dataset_configs"], desc="Processing languages"):
        dataset, lang = dataset_config.split(":")
        print(dataset, lang)
        normalized_lang = lang_choices_to_qualified_name[lang]

        input_dir = args["in_dir"] / args["in_path"] / dataset
        # Create filename with appropriate suffixes
        filename_parts = [lang]
        if args["romanized"]:
            filename_parts.append("romanized")
        if args["shuffle_words"]:
            filename_parts.append("shuffled")
        filename = "_".join(filename_parts) + ".pt"
        file_path = input_dir / filename

        # Debug print to show status
        status_parts = []
        if args["romanized"]:
            status_parts.append("romanized")
        if args["shuffle_words"]:
            status_parts.append("shuffled")
        status = "_".join(status_parts) if status_parts else "normal"
        print(f"[DEBUG] Reading {status} activations from: {file_path}")

        activations_count = torch.load(file_path)
        aggregate_activations_count(
            lang_to_activations_count, normalized_lang, activations_count
        )

    # LAPE Input
    sorted_lang = sorted(lang_to_activations_count.keys())
    (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    ) = stack_activations_count(lang_to_activations_count, sorted_lang)

    logger.info(
        f"num_examples: {num_examples.tolist()}\n\t   num_tokens: {num_tokens.tolist()}"
    )

    # LAPE Output
    if args["algorithm"] == "sae_lape":
        (
            final_indice,
            final_indice_global_max_active,
            final_indice_global_min_active,
            final_indice_global_avg_active,
            features_info,
        ) = sae_lape(
            num_examples,
            num_tokens,
            over_zero_token,
            over_zero_example,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
            args["topk_threshold_ratio"],
            args["example_rate"],
            args["top"],
            args["top_per_layer"],
            sorted_lang,
            args["entropy_threshold"],
            args["lang_specific"],
            args["lang_shared"],
            args["shared_count"],
            args["top_by_frequency"],
        )
    elif args["algorithm"] == "lape":
        (
            final_indice,
            final_indice_global_max_active,
            final_indice_global_min_active,
            final_indice_global_avg_active,
            features_info,
        ) = lape(
            over_zero_token,
            num_tokens,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
            sorted_lang,
        )

    logger.info("Save SAE features")

    sae_features = (
        {
            "sae_features": get_sae_features(
                final_indice,
                args["model"],
                args["sae_model"],
                args["layers"],
                args["local_sae_dir"],
            )
        }
        if args["sae_model"]
        else {}
    )

    # Save as a sparse tensor
    global_min_active_over_zero[global_min_active_over_zero == math.inf] = 0

    output = {
        "final_indice": final_indice,
        "final_indice_global_max_active": final_indice_global_max_active,
        "final_indice_global_min_active": final_indice_global_min_active,
        "final_indice_global_avg_active": final_indice_global_avg_active,
        "sorted_lang": sorted_lang,
        "num_examples": num_examples,
        "num_tokens": num_tokens,
        "over_zero_token": over_zero_token.to_sparse(),
        "over_zero_example": over_zero_example.to_sparse(),
        "features_info": features_info,
        **sae_features,
    }

    out_dir = args["out_dir"] / args["out_path"]
    out_filename = args["out_filename"]

    # If out_filename is a Path, use its stem/suffix; otherwise coerce to Path-like pieces
    if isinstance(out_filename, Path):
        base_name = out_filename.stem
        extension = out_filename.suffix  # includes leading dot or empty
    else:
        out_filename = str(out_filename)
        base_name, dot, rest = out_filename.partition(".")
        extension = f".{rest}" if dot else ""

    # Add suffixes to filename based on flags
    if args["romanized"] or args["shuffle_words"]:
        suffixes = []
        if args["romanized"]:
            suffixes.append("romanized")
        if args["shuffle_words"]:
            suffixes.append("shuffled")
        suffix_str = "_".join(suffixes)
        base_name = f"{base_name}_{suffix_str}"

    filename = f"{base_name}{extension}"
    file_path = out_dir / filename

    os.makedirs(out_dir, exist_ok=True)

    torch.save(output, file_path)


if __name__ == "__main__":
    set_deterministic()

    logger = TqdmLoggingHandler.get_logger("identify")

    args = parse_args()
    main(args)
