import torch
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import paths

def make_csv(path):
    # Path to your .pt file
    file_name=path.split('/')[-1].split('.')[0]
    print(file_name)

    # Load file
    data = torch.load(path, map_location="cpu")

    # Extract the nested dict
    features_info = data["features_info"]

    rows = []

    for lang, info in features_info.items():

        if file_name.endswith("all") or file_name.endswith("all_shuffled"):
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


    # Create DataFrame
    df = pd.DataFrame(rows)

    # Define base output directory
    output_base = os.path.join(paths.PROJECT_ROOT, "language-specific-features/output_romanization_expt_diacritics/csvs")
    os.makedirs(output_base, exist_ok=True)
    output_file = os.path.join(output_base, f"features_layerwise_{file_name}.csv")

    # Sort and save
    if file_name.endswith("all") or file_name.endswith("all_shuffled"):
        df = df.sort_values(by=["language", "layer", "selected_prob"], ascending=[True, True, False])
    else:
        df = df.sort_values(by=["language", "layer", "entropies"], ascending=[True, True, True])
    df.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df)} rows to {output_file}")
    df.head(10)

if __name__ == "__main__":
    base_data_dir = os.path.join(paths.PROJECT_ROOT, "language-specific-features/output_romanization_expt_diacritics")
    
    paths_to_process = [
        os.path.join(base_data_dir, "sae_features_specific/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all_romanized.pt"),
        os.path.join(base_data_dir, "sae_features_specific/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt")
    ]
    
    for i in range(2,11):
        paths_to_process.append(os.path.join(base_data_dir, f"sae_features_shared/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_shared_{i}.pt"))
        paths_to_process.append(os.path.join(base_data_dir, f"sae_features_shared/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_shared_{i}_romanized.pt"))
    
    for path in paths_to_process:
        if os.path.exists(path):
            make_csv(path)
        else:
            print(f"File not found: {path}")