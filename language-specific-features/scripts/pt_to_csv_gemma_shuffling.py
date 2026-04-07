import torch
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to sys.path to find utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import paths

def make_csv(path, output_dir):
    # Path to your .pt file
    file_name = Path(path).stem
    if file_name == "lape_neuron":
        file_name = "lape_all"
    elif file_name == "lape_neuron_shuffled":
        file_name = "lape_all_shuffled"
    
    print(f"Processing {file_name} from {path}")

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
    df = pd.DataFrame(rows)

    output_file = output_dir / f"features_layerwise_{file_name}.csv"

    # Sort and save
    if file_name.endswith("all") or file_name.endswith("all_shuffled"):
        df = df.sort_values(by=["language", "layer", "selected_prob"], ascending=[True, True, False])
    else:
        df = df.sort_values(by=["language", "layer", "entropies"], ascending=[True, True, True])
    df.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    base_dir = Path(paths.PROJECT_ROOT) / "language-specific-features/output_lape_shuffling_gemma"
    output_dir = base_dir / "output_lape_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = [
        base_dir / "mlp_acts_specific/gemma-2-2b/lape_neuron.pt",
        base_dir / "mlp_acts_specific/gemma-2-2b/lape_neuron_shuffled.pt"
    ]
    
    for i in range(2, 16):
        shared_path = base_dir / f"mlp_acts_shared/gemma-2-2b/lape_shared_{i}.pt"
        if shared_path.exists():
            paths.append(shared_path)
        shared_shuf_path = base_dir / f"mlp_acts_shared/gemma-2-2b/lape_shared_{i}_shuffled.pt"
        if shared_shuf_path.exists():
            paths.append(shared_shuf_path)
    
    for path in paths:
        if path.exists():
            make_csv(str(path), output_dir)
        else:
            print(f"File not found: {path}")

