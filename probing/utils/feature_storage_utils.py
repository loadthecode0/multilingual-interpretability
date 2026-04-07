import torch
import pandas as pd
from pathlib import Path


def save_sae_lape_features(
    final_indices,
    features_info,
    shared_features,
    sorted_langs,
    model_name,
    layer_names,
    dataset,
    split,
    method="sae_lape",
    base_dir="identification",
    top_k=100,
    experiment_tag=""
):
    """Save SAE-LAPE results to CSV files with proper layer naming.
       Also store an extra CSV for shared neurons across languages.
    """
    saved_files = []
    coord_lang_map = {}  # (layer, feat) -> [languages]
    # print(shared_features)

    for lang_idx, lang in enumerate(sorted_langs):
        if lang_idx >= len(final_indices) or lang not in features_info:
            continue
            
        lang_indices = final_indices[lang_idx]
        lang_info = features_info[lang]
        
        # Create mapping from (layer, feature) to entropy/prob
        if method == "sae_lape":
            coord_to_data = {}
            for i, (layer_idx, feat_idx) in enumerate(lang_info['indices']):
                coord_to_data[(layer_idx, feat_idx)] = (
                    lang_info['entropies'][i].item(),
                    lang_info['selected_probs'][i].item()
                )
        elif method == "magnitude":
            coord_to_data = {}
            for i, (layer_idx, feat_idx) in enumerate(lang_info['indices']):
                coord_to_data[(layer_idx, feat_idx)] = (
                    lang_info['avg_activations'][i].item()
                )
        
        # Save features for each layer using actual layer names
        for layer_idx, layer_features in enumerate(lang_indices):
            if len(layer_features) == 0:
                continue
            
            if layer_idx < len(layer_names):
                layer_name = layer_names[layer_idx]
                layer_num = layer_name.split('.')[1] if '.' in layer_name else str(layer_idx)
            else:
                layer_num = str(layer_idx)
                
            data = []
            for feat_idx in layer_features:
                coord = (layer_idx, feat_idx.item())
                
                # Build normal CSV row
                if coord in coord_to_data:
                    if method == "sae_lape":
                        entropy, prob = coord_to_data[coord]
                        data.append({
                            'feature_idx': feat_idx.item(),
                            'entropy': entropy,
                            'activation_prob': prob,
                            'rank': len(data) + 1
                        })
                    elif method == "magnitude":
                        avg_act = coord_to_data[coord]
                        data.append({
                            'feature_idx': feat_idx.item(),
                            'avg_activation': avg_act
                        })
                
                # Track shared neurons
                coord_lang_map.setdefault(coord, []).append(lang)
            
            if data:
                dataset_folder = f"{dataset}"
                if experiment_tag:
                    dataset_folder += f"-{experiment_tag}"
                dir_path = Path(base_dir) / model_name / method / f"layer_{layer_num}" / dataset_folder / split
                dir_path.mkdir(parents=True, exist_ok=True)
                
                df = pd.DataFrame(data[:top_k])
                csv_path = dir_path / f"{lang}.csv"
                df.to_csv(csv_path, index=False)
                saved_files.append(csv_path)

    # ----- NEW PART: Save shared neurons CSV -----
    if method == "sae_lape" and shared_features:
        print("save:", shared_features)
        dataset_folder = f"{dataset}"
        if experiment_tag:
            dataset_folder += f"-{experiment_tag}"
        dir_path = Path(base_dir) / model_name / method / f"layer_{layer_num}" / dataset_folder / split
        dir_path.mkdir(parents=True, exist_ok=True)

        df_shared = pd.DataFrame(shared_features)
        df_shared["languages"] = df_shared["languages"].apply(lambda x: ",".join(x))
        csv_path = dir_path / "shared_neurons.csv"
        df_shared.to_csv(csv_path, index=False)
        print(f"[INFO] Saved shared neurons CSV: {csv_path}")


    print(f"Saved {len(saved_files)} files to {base_dir}/{model_name}/{method}/")