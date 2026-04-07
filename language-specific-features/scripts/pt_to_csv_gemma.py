import torch
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import paths

def make_csv(path, output_dir):
    # Path to your .pt file
    file_name = Path(path).stem
    print(f"Processing {file_name}...")

    # Load file
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        return

    # Extract the nested dict
    features_info = data.get("features_info", {})
    if not features_info:
        print(f"⚠️ No features_info found in {path}")
        return

    rows = []

    for lang, info in features_info.items():
        if file_name.endswith("all") or file_name.endswith("all_shuffled"):
            indices = info.get("indicies", []) # Note the spelling 'indicies' from original script
            probs = info.get("selected_probs", None)
            ents = info.get("entropies", None)

            # Convert tensors to lists if needed
            if torch.is_tensor(probs):
                probs = probs.tolist()
            if torch.is_tensor(ents):
                ents = ents.tolist()

            # Zip all together safely
            for i, (layer, feat_idx) in enumerate(indices):
                prob = probs[i] if probs and i < len(probs) else None
                ent = ents[i] if ents and i < len(ents) else None
                rows.append({
                    "language": lang,
                    "layer": layer,
                    "sae_feature_idx": feat_idx,
                    "selected_prob": prob,
                    "entropy": ent
                })
        else:
            indices = info.get("indicies", [])
            ents = info.get("entropies", None)

            # Convert tensors to lists if needed
            if torch.is_tensor(ents):
                ents = ents.tolist()

            # Zip all together safely
            for i, (layer, feat_idx) in enumerate(indices):
                ent = ents[i] if ents and i < len(ents) else None
                rows.append({
                    "language": lang,
                    "layer": layer,
                    "sae_feature_idx": feat_idx,
                    "entropies": ent
                })

    if not rows:
        print(f"⚠️ No rows produced for {file_name}")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"features_layerwise_{file_name}.csv")

    # Sort and save
    if file_name.endswith("all") or file_name.endswith("all_shuffled"):
        if "selected_prob" in df.columns:
            df = df.sort_values(by=["language", "layer", "selected_prob"], ascending=[True, True, False])
    else:
        if "entropies" in df.columns:
            df = df.sort_values(by=["language", "layer", "entropies"], ascending=[True, True, True])
    
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    base_out = os.path.join(paths.PROJECT_ROOT, "language-specific-features/output_shuffling_gemma")
    model_part = "gemma-2-2b/gemma-scope-2b-pt-mlp-canonical"
    
    specific_dir = os.path.join(base_out, "sae_features_specific", model_part)
    shared_dir = os.path.join(base_out, "sae_features_shared", model_part)
    csv_out_dir = os.path.join(base_out, "csvs")
    
    paths = []
    if os.path.exists(specific_dir):
        for f in os.listdir(specific_dir):
            if f.endswith(".pt"):
                paths.append(os.path.join(specific_dir, f))
                
    if os.path.exists(shared_dir):
        for f in os.listdir(shared_dir):
            if f.endswith(".pt"):
                paths.append(os.path.join(shared_dir, f))
                
    for path in paths:
        make_csv(path, csv_out_dir)

