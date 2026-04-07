#!/usr/bin/env python3
"""
Compute dominance scores for SAE / raw neurons based on lang2vec probing R² CSVs.

Dominance(n, family) =
    max R²(n, family columns) - max R²(n, other family columns)

Outputs ranked neuron lists per layer and per family.

Stops at Step 4 (no label-randomization baseline).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


# -----------------------------
# Feature family definitions
# -----------------------------
FAMILIES: Dict[str, List[str]] = {
    "syntax": [
        "syntax_wals", "syntax_ethnologue", "syntax_sswl",
        "syntax_average", "syntax_knn",
    ],
    "phonology": [
        "phonology_wals", "phonology_ethnologue",
        "phonology_average", "phonology_knn",
    ],
    "inventory": [
        "inventory_phoible_ph", "inventory_phoible_spa",
        "inventory_phoible_upsid", "inventory_knn",
        "inventory_average",
    ],
    "control": ["fam", "geo"],
}


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dominance rankings for neurons.")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--scores-subdir", type=str, default="scores")
    parser.add_argument("--output-subdir", type=str, default="dominance")

    # selection options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topk", type=int, help="Select top-K neurons per family.")
    group.add_argument("--top-percent", type=float, help="Select top-X percent neurons.")

    parser.add_argument(
        "--min-max-r2",
        type=float,
        default=0.0,
        help="Drop neurons whose maximum R² across all families is below this value.",
    )

    return parser.parse_args()


# -----------------------------
# Core logic
# -----------------------------
def compute_family_scores(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return max R² per neuron for each family."""
    family_scores = {}
    for fam, cols in FAMILIES.items():
        existing = [c for c in cols if c in df.columns]
        if not existing:
            continue
        family_scores[fam] = df[existing].max(axis=1)
    return family_scores


def compute_dominance(
    df: pd.DataFrame,
    family_scores: Dict[str, pd.Series],
    target_family: str,
) -> pd.DataFrame:
    """Compute dominance table for a single family."""
    target = family_scores[target_family]

    other_fams = [f for f in family_scores if f != target_family]
    max_other = pd.concat(
        [family_scores[f] for f in other_fams], axis=1
    ).max(axis=1)

    out = pd.DataFrame({
        "neuron_idx": df["neuron_idx"],
        "target_family_score": target,
        "max_other_family_score": max_other,
    })
    out["dominance"] = out["target_family_score"] - out["max_other_family_score"]
    out = out.sort_values("dominance", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def select_top(df: pd.DataFrame, topk: int | None, top_percent: float | None) -> pd.DataFrame:
    if topk is not None:
        return df.head(topk)
    assert top_percent is not None
    k = max(1, int(len(df) * top_percent / 100.0))
    return df.head(k)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    scores_root = base_dir / args.scores_subdir
    out_root = base_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(p for p in scores_root.iterdir() if p.is_dir()):
        print(f"[INFO] Processing {exp_dir.name}")
        exp_out = out_root / exp_dir.name
        exp_out.mkdir(parents=True, exist_ok=True)

        for layer_csv in sorted(exp_dir.glob("layer_*.csv")):
            layer = layer_csv.stem
            layer_out = exp_out / layer
            layer_out.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(layer_csv)

            # Drop neuron_idx-less rows just in case
            if "neuron_idx" not in df.columns:
                continue

            family_scores = compute_family_scores(df)
            if not family_scores:
                continue

            # Optional global filter
            max_r2_all = pd.concat(list(family_scores.values()), axis=1).max(axis=1)
            df = df.loc[max_r2_all >= args.min_max_r2].reset_index(drop=True)
            for fam in family_scores:
                family_scores[fam] = family_scores[fam].loc[df.index]

            for fam in FAMILIES:
                if fam not in family_scores:
                    continue
                dom_df = compute_dominance(df, family_scores, fam)
                dom_df = select_top(dom_df, args.topk, args.top_percent)

                out_path = layer_out / f"dominance_{fam}.csv"
                dom_df.to_csv(out_path, index=False)

            # --- New: Top R2 across all categories ---
            # Using the re-filtered max_r2_all
            top_r2_df = pd.DataFrame({
                "neuron_idx": df["neuron_idx"],
                "max_r2": pd.concat(list(family_scores.values()), axis=1).max(axis=1)
            }).sort_values("max_r2", ascending=False).head(200).reset_index(drop=True)
            top_r2_df["rank"] = top_r2_df.index + 1
            top_r2_df.to_csv(layer_out / "dominance_top_r2_all.csv", index=False)

        print(f"[OK] Wrote dominance results → {exp_out}")


if __name__ == "__main__":
    main()
