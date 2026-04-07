import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from bracex import expand
from const import (
    dataset_choices,
    lang_choices,
    lang_choices_to_qualified_name,
    mlp_to_index,
    model_choices,
    sae_model_choices,
)
from loader import load_activations
from tqdm.auto import tqdm
from utils import (
    TqdmLoggingHandler,
    get_model_dirname,
    get_project_dir,
    set_deterministic,
)


class Args(TypedDict):
    model: str
    dataset: str
    languages: list[str]
    layers: list[str]
    sae_model: str
    in_dir: Path
    out_dir: Path
    shuffle_words: bool
    romanized: bool
    cross_romanized: bool
    native_output_root: Path | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Visualize sae features from a dataset for particular layers and languages."
    )

    parser.add_argument(
        "model",
        help="model name",
        type=str,
        # choices=model_choices,
    )

    parser.add_argument(
        "dataset",
        help="dataset name",
        type=str,
        choices=dataset_choices,
    )

    parser.add_argument(
        "--lang",
        help="language(s) to be processed",
        type=str,
        default=[],
        nargs="+",
        choices=lang_choices,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--sae-model",
        help="sae model name",
        type=str,
        default=None,
        choices=sae_model_choices,
    )

    parser.add_argument(
        "--in-dir",
        help="input directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
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
    parser.add_argument(
        "--cross-romanized",
        help="When using romanized data, keep stats only for native-selected features.",
        action="store_true",
    )
    parser.add_argument(
        "--native-output-root",
        help="Root directory containing native lape outputs (defaults to --out-dir).",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    return {
        "model": args.model,
        "dataset": args.dataset,
        "languages": args.lang,
        "layers": processed_layers,
        "sae_model": args.sae_model,
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
        "shuffle_words": args.shuffle_words,
        "romanized": args.romanized,
        "cross_romanized": args.cross_romanized,
        "native_output_root": args.native_output_root,
    }


def load_allowed_features(native_root: Path, model: str, sae_model: str):
    """
    Build mapping {qualified_lang: {layer_name: set(feature_idx)}} based on native lape outputs.
    """

    idx_to_layer = {idx: layer for layer, idx in mlp_to_index.items()}
    lang_to_layer_features: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

    def ingest_lape(path: Path):
        if not path.exists():
            return
        data = torch.load(path, map_location="cpu")
        for lang, info in data["features_info"].items():
            for (layer_idx, feature_idx) in info["indicies"]:
                layer_name = idx_to_layer.get(layer_idx)
                if layer_name is None:
                    continue
                lang_to_layer_features[lang][layer_name].add(int(feature_idx))

    model_dir = get_model_dirname(model)

    specific_dir = native_root / "sae_features_specific" / model_dir / sae_model
    ingest_lape(specific_dir / "lape_all.pt")

    shared_dir = native_root / "sae_features_shared" / model_dir / sae_model
    if shared_dir.exists():
        for file_path in shared_dir.glob("lape_shared_*.pt"):
            if file_path.stem.endswith("_romanized"):
                continue
            ingest_lape(file_path)

    return lang_to_layer_features


def extract_features(sae_model: str, sae_features: any):
    """
    Return (top_acts, top_indices) no matter which SAE family produced the features.
    Gemma scope saves EncoderOutput objects just like EleutherAI SAEs, so we can rely
    on attribute presence rather than model name.
    """
    top_acts = getattr(sae_features, "top_acts", None)
    top_indices = getattr(sae_features, "top_indices", None)

    if top_acts is not None and top_indices is not None:
        return top_acts, top_indices

    if isinstance(sae_features, dict):
        acts = sae_features.get("top_acts")
        indices = sae_features.get("top_indices")
        if acts is not None and indices is not None:
            return acts, indices

    raise TypeError(
        "SAE feature object must expose `top_acts` and `top_indices`. "
        f"Got type: {type(sae_features)} with attrs: {dir(sae_features)}"
    )


def process_sae_features(
    sae_features_list: list[any],
    sae_model: str,
    layer: str,
    lang: str,
    rounding_digit=3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sae_feature_index_to_activations = defaultdict(list)
    sae_feature_index_to_dataset_id_token_id_act_val = defaultdict(list)

    # print(sae_features_list)
    for dataset_row_index, sae_features in enumerate(sae_features_list):
        top_acts, top_indices = extract_features(sae_model, sae_features)
        top_act_index_per_token = zip(top_acts.squeeze(0), top_indices.squeeze(0))

        for token_index, (top_act, top_index) in enumerate(top_act_index_per_token):
            for act_val, feature_index in zip(top_act.tolist(), top_index.tolist()):
                sae_feature_index_to_activations[feature_index].append(act_val)
                dataset_id_token_id_act_val = (
                    dataset_row_index,
                    token_index,
                    round(act_val, rounding_digit),
                )
                sae_feature_index_to_dataset_id_token_id_act_val[feature_index].append(
                    dataset_id_token_id_act_val
                )

    sae_features_count = {}
    sae_features_avg = {}
    sae_features_q1 = {}
    sae_features_median = {}
    sae_features_q3 = {}
    sae_features_min_active = {}
    sae_features_max_active = {}
    sae_features_std = {}

    for feature_index, activations in sae_feature_index_to_activations.items():
        sae_features_count[feature_index] = len(activations)
        sae_features_avg[feature_index] = round(
            np.mean(activations).item(), rounding_digit
        )
        sae_features_q1[feature_index] = round(
            np.percentile(activations, 25).item(), rounding_digit
        )
        sae_features_median[feature_index] = round(
            np.median(activations).item(), rounding_digit
        )
        sae_features_q3[feature_index] = round(
            np.percentile(activations, 75).item(), rounding_digit
        )
        sae_features_min_active[feature_index] = round(
            np.min(activations).item(), rounding_digit
        )
        sae_features_max_active[feature_index] = round(
            np.max(activations).item(), rounding_digit
        )
        sae_features_std[feature_index] = round(
            np.std(activations).item(), rounding_digit
        )

    # Create a dataframe from the statistics
    statistics = {
        "count": sae_features_count,
        "avg": sae_features_avg,
        "q1": sae_features_q1,
        "median": sae_features_median,
        "q3": sae_features_q3,
        "min_active": sae_features_min_active,
        "max_active": sae_features_max_active,
        "std": sae_features_std,
        "lang": lang,
        "layer": layer,
    }

    df_statistics = pd.DataFrame(statistics)
    df_statistics.sort_index(inplace=True)
    df_statistics.reset_index(inplace=True)

    # Create a dataframe from the dataset_token_activations
    dataset_token_activations = {
        "count": sae_features_count,
        "dataset_row_id_token_id_act_val": sae_feature_index_to_dataset_id_token_id_act_val,
    }

    df_dataset_token_activations = pd.DataFrame(dataset_token_activations)
    df_dataset_token_activations.sort_index(inplace=True)
    df_dataset_token_activations.reset_index(inplace=True)

    return df_statistics, df_dataset_token_activations


def main(args: Args, allowed_features: dict[str, dict[str, set[int]]]):
    model_dir = get_model_dirname(args["model"])

    for lang in tqdm(args["languages"], desc="Processing languages"):
        for layer in tqdm(args["layers"], desc="Processing layers", leave=False):
            # Create input directory with appropriate suffixes
            lang_suffix_parts = [lang]
            if args["romanized"]:
                lang_suffix_parts.append("romanized")
            if args["shuffle_words"]:
                lang_suffix_parts.append("shuffled")
            lang_suffix = "_".join(lang_suffix_parts)
            
            input_dir = (
                args["in_dir"]
                / "sae_features"
                / model_dir
                / args["sae_model"]
                / args["dataset"]
                / lang_suffix
            )

            # Debug print to show status
            status_parts = []
            if args["romanized"]:
                status_parts.append("romanized")
            if args["shuffle_words"]:
                status_parts.append("shuffled")
            status = "_".join(status_parts) if status_parts else "normal"
            print(f"[DEBUG] Reading {status} SAE features from: {input_dir}")

            sae_features = load_activations(input_dir, layer, logger)
            df_statistics, df_dataset_token_activations = process_sae_features(
                sae_features, args["sae_model"], layer, lang
            )

            if args["cross_romanized"]:
                qualified_lang = lang_choices_to_qualified_name.get(lang, lang)
                allowed_layers = allowed_features.get(qualified_lang, {})
                allowed_indices = allowed_layers.get(layer, set())

                if not allowed_indices:
                    continue

                df_statistics = df_statistics[df_statistics["index"].isin(allowed_indices)]
                df_dataset_token_activations = df_dataset_token_activations[
                    df_dataset_token_activations["index"].isin(allowed_indices)
                ]

                if df_statistics.empty:
                    print(f"No statistics for {lang} and {layer}")
                    continue

            # Save the statistics
            # Create output directory with appropriate suffixes
            summary_suffix_parts = ["summary"]
            if args["romanized"]:
                summary_suffix_parts.append("romanized")
            if args["shuffle_words"]:
                summary_suffix_parts.append("shuffled")
            summary_suffix = "_".join(summary_suffix_parts)
            
            output_dir = (
                args["out_dir"]
                / "statistics"
                / model_dir
                / args["sae_model"]
                / args["dataset"]
                / summary_suffix
                / layer
            )
            os.makedirs(output_dir, exist_ok=True)

            df_statistics.to_csv(output_dir / f"{lang}.csv", index=False)

            # Save the dataset_token_activations
            # Create output directory with appropriate suffixes
            token_suffix_parts = ["dataset_token_activations"]
            if args["romanized"]:
                token_suffix_parts.append("romanized")
            if args["shuffle_words"]:
                token_suffix_parts.append("shuffled")
            token_suffix = "_".join(token_suffix_parts)
            
            output_dir = (
                args["out_dir"]
                / "statistics"
                / model_dir
                / args["sae_model"]
                / args["dataset"]
                / token_suffix
                / layer
            )
            os.makedirs(output_dir, exist_ok=True)

            df_dataset_token_activations.to_csv(output_dir / f"{lang}.csv", index=False)


if __name__ == "__main__":
    set_deterministic()

    logger = TqdmLoggingHandler.get_logger("statistics")

    args = parse_args()

    allowed_features = {}
    if args["cross_romanized"]:
        native_root = (
            args["native_output_root"]
            if args["native_output_root"] is not None
            else args["out_dir"]
        )
        allowed_features = load_allowed_features(
            native_root, args["model"], args["sae_model"]
        )
        if not allowed_features:
            logger.warning(
                "Cross romanized flag enabled but no native features were found. No stats will be written."
            )
    main(args, allowed_features)
