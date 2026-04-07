#!/usr/bin/env python3
"""
Precompute layer-wise aggregated statistics for lang2vec probing scores.

This script reads the neuron-level max_r2 CSV files stored under `scores/`
and writes much smaller per-layer CSVs (feature, avg, max) under `agg_scores/`
so the plotting script can run quickly without re-aggregating every time.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate neuron-wise layer scores into compact CSVs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory that contains the scores/ folder.",
    )
    parser.add_argument(
        "--scores-subdir",
        type=str,
        default="scores",
        help="Relative path from base-dir to the raw neuron-wise score CSVs.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="agg_scores",
        help="Relative path (from base-dir) where aggregated CSVs will be stored.",
    )
    return parser.parse_args()


def collect_layer_frames(result_dir: Path) -> Tuple[List[Tuple[int, pd.DataFrame, List[str]]], List[str]]:
    """Load all layer CSVs and determine the common feature order."""
    layer_files = sorted(
        result_dir.glob("layer_*.csv"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not layer_files:
        return [], []

    parsed_layers: List[Tuple[int, pd.DataFrame, List[str]]] = []
    for layer_file in layer_files:
        layer_num = int(layer_file.stem.split("_")[1])
        df = pd.read_csv(layer_file)
        feature_cols = [col for col in df.columns if col != "neuron_idx"]
        parsed_layers.append((layer_num, df, feature_cols))

    common_features = set(parsed_layers[0][2])
    for _, _, features in parsed_layers[1:]:
        common_features &= set(features)

    if not common_features:
        return parsed_layers, []

    feature_order = [feat for feat in parsed_layers[0][2] if feat in common_features]
    return parsed_layers, feature_order


def aggregate_result_dir(result_dir: Path, output_dir: Path) -> None:
    parsed_layers, feature_order = collect_layer_frames(result_dir)

    if not parsed_layers:
        print(f"No CSV files found under {result_dir}, skipping.")
        return

    if not feature_order:
        print(
            f"Skipping {result_dir}: no common features were found across layers."
        )
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features_present = set().union(*(features for _, _, features in parsed_layers))
    dropped_features = sorted(all_features_present - set(feature_order))
    if dropped_features:
        print(
            f"{result_dir}: Dropped inconsistent features: {dropped_features}"
        )

    for layer_num, df, _ in parsed_layers:
        stats_df = pd.DataFrame(
            {
                "feature": feature_order,
                "avg": df[feature_order].mean().values,
                "max": df[feature_order].max().values,
            }
        )
        stats_df.to_csv(output_dir / f"layer_{layer_num}.csv", index=False)

    print(f"Wrote aggregated scores to {output_dir}")


def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir
    scores_dir = base_dir / args.scores_subdir
    output_root = base_dir / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    for result_dir in sorted(p for p in scores_dir.iterdir() if p.is_dir()):
        result_output_dir = output_root / result_dir.name
        aggregate_result_dir(result_dir, result_output_dir)


if __name__ == "__main__":
    main()

