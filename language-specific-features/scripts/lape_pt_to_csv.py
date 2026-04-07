#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch


def lape_pt_to_csv(pt_path: Path, out_path: Path):
    """
    Convert a lape_neuron.pt (or similar) file into a CSV listing language,
    layer, neuron index, and optional scalar statistics.
    """
    data = torch.load(pt_path, map_location="cpu")
    features_info = data.get("features_info")
    if not isinstance(features_info, dict):
        raise ValueError(f"'features_info' not found in {pt_path}")

    rows = []
    for lang, info in features_info.items():
        indices = info.get("indicies", [])
        selected_probs = info.get("selected_probs")
        entropies = info.get("entropies")

        if torch.is_tensor(selected_probs):
            selected_probs = selected_probs.tolist()
        if torch.is_tensor(entropies):
            entropies = entropies.tolist()

        for i, (layer, neuron_idx) in enumerate(indices):
            row = {
                "language": lang,
                "layer": layer,
                "neuron_idx": neuron_idx,
            }
            if selected_probs is not None and i < len(selected_probs):
                row["selected_prob"] = selected_probs[i]
            if entropies is not None and i < len(entropies):
                row["entropy"] = entropies[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(by=["language", "layer", "neuron_idx"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert lape_neuron.pt outputs into CSV files."
    )
    parser.add_argument(
        "--native-pt",
        type=Path,
        required=True,
        help="Path to lape_neuron.pt from the native run.",
    )
    parser.add_argument(
        "--romanized-pt",
        type=Path,
        help="Path to lape_neuron.pt from the romanized run (optional).",
    )
    parser.add_argument(
        "--shuffled-pt",
        type=Path,
        help="Path to lape_neuron_shuffled.pt from the shuffling run (optional).",
    )
    parser.add_argument(
        "--shared-pt",
        type=Path,
        nargs="*",
        help="Optional list of lape_shared_*.pt files to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output_lape_csvs"),
        help="Directory to write CSV files into.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    native_csv = args.output_dir / "lape_neuron_native.csv"
    lape_pt_to_csv(args.native_pt, native_csv)

    if args.romanized_pt:
        roman_csv = args.output_dir / "lape_neuron_romanized.csv"
        lape_pt_to_csv(args.romanized_pt, roman_csv)

    if args.shuffled_pt:
        shuffled_csv = args.output_dir / "lape_neuron_shuffled.csv"
        lape_pt_to_csv(args.shuffled_pt, shuffled_csv)

    if args.shared_pt:
        for shared_path in args.shared_pt:
            shared_name = shared_path.stem
            shared_csv = args.output_dir / f"{shared_name}.csv"
            lape_pt_to_csv(shared_path, shared_csv)


if __name__ == "__main__":
    main()
