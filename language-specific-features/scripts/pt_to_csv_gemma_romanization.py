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
    # Standardize naming for downstream scripts
    if file_name == "lape_all":
        pass
    elif file_name == "lape_all_romanized":
        pass
    
    print(f"Processing {file_name} from {path}")

    # Load file
    data = torch.load(path, map_location="cpu")

    # Extract the nested dict
    features_info = data["features_info"]

    rows = []

    for lang, info in features_info.items():

        if "lape_all" in file_name:
            indices = info.get("indicies", [])
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
                    "neuron_idx": feat_idx,
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
                    "neuron_idx": feat_idx,
                    "entropies": ent
                })


    # Create DataFrame
    if not rows:
        print(f"⚠️ No data found in {file_name}, skipping CSV generation.")
        return

    df = pd.DataFrame(rows)

    output_file = output_dir / f"features_layerwise_{file_name}.csv"

    # Sort and save
    if "lape_all" in file_name:
        df = df.sort_values(by=["language", "layer", "selected_prob"], ascending=[True, True, False])
    else:
        df = df.sort_values(by=["language", "layer", "entropies"], ascending=[True, True, True])
    df.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    base_dir = Path(paths.PROJECT_ROOT) / "language-specific-features/output_romanization_expt_gemma"
    # For downstream compare_lape_runs.py, it expects a 'csvs' subdirectory
    output_dir = base_dir / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sae_model_path = "gemma-2-2b/gemma-scope-2b-pt-mlp-canonical"
    
    paths = [
        base_dir / "sae_features_specific" / sae_model_path / "lape_all.pt",
        base_dir / "sae_features_specific" / sae_model_path / "lape_all_romanized.pt"
    ]
    
    for i in range(2, 16):
        shared_path = base_dir / "sae_features_shared" / sae_model_path / f"lape_shared_{i}.pt"
        if shared_path.exists():
            paths.append(shared_path)
        shared_rom_path = base_dir / "sae_features_shared" / sae_model_path / f"lape_shared_{i}_romanized.pt"
        if shared_rom_path.exists():
            paths.append(shared_rom_path)
    
    for path in paths:
        if path.exists():
            make_csv(str(path), output_dir)
        else:
            print(f"File not found: {path}")

