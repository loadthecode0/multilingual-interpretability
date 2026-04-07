import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from const import (
    lang_choices_to_iso639_1,
    lang_choices_to_qualified_name,
    mlp_to_index,
)


def build_inverse_lang_maps():
    qualified_to_choice = {}
    for choice, qualified in lang_choices_to_qualified_name.items():
        qualified_to_choice[qualified] = choice
    return qualified_to_choice


def build_index_to_layer():
    # Use only MLP hookpoints (the ones actually collected) to avoid act_fn collisions
    return {idx: layer for layer, idx in mlp_to_index.items()}


def load_lape_features(lape_path: Path, idx_to_layer: dict[int, str]):
    data = torch.load(lape_path, map_location="cpu")
    lang_to_layer_features = defaultdict(lambda: defaultdict(list))

    for lang, info in data["features_info"].items():
        indices = info["indicies"]
        probs = info["selected_probs"]
        entropies = info["entropies"]

        for (layer_idx, feature_idx), prob, entropy in zip(indices, probs, entropies):
            layer_name = idx_to_layer.get(layer_idx)
            if layer_name is None:
                continue
            lang_to_layer_features[lang][layer_name].append(
                {
                    "feature_idx": int(feature_idx),
                    "native_selected_prob": float(prob),
                    "native_entropy": float(entropy),
                }
            )

    return lang_to_layer_features


def write_cross_analysis(
    target_path: Path,
    features: dict,
    summary_dir: Path,
    qualified_to_choice: dict,
):
    csv_rows = []

    @lru_cache(maxsize=None)
    def load_summary(layer_name: str, lang_choice: str):
        csv_path = summary_dir / layer_name / f"{lang_choice}.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)

    for qualified_lang, layer_map in features.items():
        lang_choice = qualified_to_choice.get(qualified_lang)
        if lang_choice is None:
            continue
        iso_code = lang_choices_to_iso639_1.get(lang_choice, lang_choice)

        for layer_name, feature_list in layer_map.items():
            df = load_summary(layer_name, lang_choice)
            if df is None:
                continue
            df_index = df.set_index("index")

            for feature in feature_list:
                feature_idx = feature["feature_idx"]
                row = df_index.loc[feature_idx] if feature_idx in df_index.index else None
                row_dict = {
                    "qualified_lang": qualified_lang,
                    "lang_choice": lang_choice,
                    "iso639_1": iso_code,
                    "layer": layer_name,
                    "sae_feature_idx": feature_idx,
                    "native_selected_prob": feature["native_selected_prob"],
                    "native_entropy": feature["native_entropy"],
                }
                if row is not None:
                    for col in row.index:
                        row_dict[f"romanized_{col}"] = row[col]
                else:
                    row_dict["romanized_missing"] = True

                csv_rows.append(row_dict)

    if not csv_rows:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(target_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-analyze romanized statistics on native-selected features."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_romanization_expt"),
        help="Root directory containing pipeline outputs.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="icu_dakshina",
        help="Dataset name used in outputs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Llama-3.2-1B",
        help="Model name segment used in paths.",
    )
    parser.add_argument(
        "--sae-model",
        type=str,
        default="EleutherAI/sae-Llama-3.2-1B-131k",
        help="SAE model name segment used in paths.",
    )
    parser.add_argument(
        "--cross-dir",
        type=str,
        default="cross_analysis",
        help="Subdirectory name for cross-analysis outputs.",
    )
    args = parser.parse_args()

    model_dir = Path(args.output_root)
    idx_to_layer = build_index_to_layer()
    qualified_to_choice = build_inverse_lang_maps()

    summary_root = (
        model_dir
        / "statistics"
        / args.model_name
        / args.sae_model
        / args.dataset
        / "summary_romanized"
    )

    cross_root = model_dir / args.cross_dir
    cross_root.mkdir(parents=True, exist_ok=True)

    # Process native lape_all selections
    lape_all_path = (
        model_dir
        / "sae_features_specific"
        / args.model_name
        / args.sae_model
        / "lape_all.pt"
    )
    if lape_all_path.exists():
        features = load_lape_features(lape_all_path, idx_to_layer)
        write_cross_analysis(
            cross_root / "lape_all_on_romanized.csv",
            features,
            summary_root,
            qualified_to_choice,
        )

    # Process native shared selections
    shared_dir = (
        model_dir
        / "sae_features_shared"
        / args.model_name
        / args.sae_model
    )
    if shared_dir.exists():
        for shared_file in sorted(shared_dir.glob("lape_shared_*.pt")):
            if shared_file.stem.endswith("_romanized"):
                continue
            features = load_lape_features(shared_file, idx_to_layer)
            target = cross_root / f"{shared_file.stem}_on_romanized.csv"
            write_cross_analysis(
                target,
                features,
                summary_root,
                qualified_to_choice,
            )


if __name__ == "__main__":
    main()


