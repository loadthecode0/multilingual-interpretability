import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm


def read_probe_csv(path):
    """Reads probe CSVs safely, adding a header if missing."""
    cols = [
        "layer", "neuron_idx", "source_languages", "num_source_langs",
        "feature_set", "feature_name", "feature_idx", "r2_score"
    ]
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None, names=cols)
        return df

    if list(df.columns) == list(range(len(cols))):
        df = pd.read_csv(path, header=None, names=cols)
    if len(df.columns) < len(cols):
        df.columns = cols[:len(df.columns)] + [f"extra_{i}" for i in range(len(df.columns) - len(cols))]
    return df


# --- CONFIG ---
mode = "llama"  # or "llama"

if mode == "gemma":
    base_dir = f"./lang2vec_probing/gemma_results"
    exp = "flores_plus"
else:
    base_dir = f"./lang2vec_probing/results"
    exp = "flores_plus-shared-flores_plus"

out_dir = f"{base_dir}/{exp}/aggregate"
os.makedirs(out_dir, exist_ok=True)

probe_files = glob(f"{base_dir}/*/{exp}/*_probes_all_neurons.csv", recursive=True)
print(f"Found {len(probe_files)} probe result files.")

# --- STEP 1: Filter valid files ---
valid_files, skipped_empty = [], []

for f in tqdm(probe_files, desc="Checking CSVs"):
    try:
        if "geo" in f:
            continue
        if os.path.getsize(f) == 0:
            skipped_empty.append((f, "empty_file"))
            continue
        df = read_probe_csv(f)
        if df.empty:
            skipped_empty.append((f, "empty_df"))
        else:
            valid_files.append(f)
    except Exception as e:
        skipped_empty.append((f, str(e)))

print(f"\n✅ {len(valid_files)} valid CSVs remain.")
if skipped_empty:
    print("\n[INFO] Skipped:")
    for f, reason in skipped_empty:
        print(f"  - {f} → {reason}")

if not valid_files:
    raise RuntimeError("No usable probe files remain.")

# --- STEP 2: Build metadata table ---
file_index = []
for f in tqdm(valid_files, desc="Building file index"):
    df = read_probe_csv(f)
    file_index.append((f, df["feature_set"].iloc[0], int(df["layer"].iloc[0])))

file_df = pd.DataFrame(file_index, columns=["path", "feature_set", "layer"])
layers = sorted(file_df["layer"].unique())
print("\nDetected layers:", layers)

# --- STEP 3: Layer-wise aggregation ---
for layer in tqdm(layers, desc="Processing layers"):
    layer_files = file_df[file_df.layer == layer]

    # Skip if nothing
    if layer_files.empty:
        continue

    summaries = []

    for _, row in layer_files.iterrows():
        fpath, feature_set = row.path, row.feature_set
        print(f"\n→ Layer {layer}, feature_set '{feature_set}'")

        df = read_probe_csv(fpath)
        if df.empty:
            print(f"   [SKIP] Empty CSV: {fpath}")
            continue

        # Group by neuron
        for neuron, rows in df.groupby("neuron_idx"):
            scores = rows["r2_score"].values
            if len(scores) == 0:
                continue

            scores = np.sort(scores)[::-1]
            n = len(scores)
            summaries.append([
                neuron, feature_set,
                scores.mean(),
                scores.max(),
                scores[:max(1, int(n * 0.25))].mean(),
                scores[:max(1, int(n * 0.50))].mean(),
                scores[:max(1, int(n * 0.75))].mean()
            ])

    if not summaries:
        print(f"   [WARN] No data for layer {layer}")
        continue

    # Combine into DataFrame
    out_df = pd.DataFrame(summaries, columns=[
        "neuron_idx", "feature_set", "avg_r2", "max_r2",
        "top25_mean", "top50_mean", "top75_mean"
    ])

    out_path = os.path.join(out_dir, f"layer_{layer}_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"   ✅ Saved {len(out_df)} rows → {out_path}")

print("\n🎯 DONE — Layerwise aggregates saved in:", out_dir)
