
import os
import pandas as pd
import re
from pathlib import Path

def get_neurons(dir_path, suffix, is_raw):
    """
    Loads all neurons from a directory matching a specific suffix.
    Returns a dict: { language: { (layer, idx): degree } }
    """
    neuron_map = {}
    idx_col = 'neuron_idx' if is_raw else 'sae_feature_idx'
    
    # Iterate through all relevant CSVs in the directory
    for f in os.listdir(dir_path):
        if not f.endswith(suffix):
            continue
        
        # Ensure we don't pick up the other condition's files
        if suffix == '.csv' and ('_romanized.csv' in f or '_shuffled.csv' in f):
            continue
            
        # Determine degree
        if 'lape_all' in f:
            degree = 1
        else:
            match = re.search(r'shared_(\d+)', f)
            if match:
                degree = int(match.group(1))
            else:
                continue # Should not happen with our filenames

        path = os.path.join(dir_path, f)
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            
            for lang in df['language'].unique():
                if lang not in neuron_map:
                    neuron_map[lang] = {}
                
                subset = df[df['language'] == lang]
                subset = subset.dropna(subset=[idx_col])
                for _, row in subset.iterrows():
                    neuron = (int(row['layer']), int(row[idx_col]))
                    # If already present, we keep the existing degree (should be same)
                    if neuron not in neuron_map[lang]:
                        neuron_map[lang][neuron] = degree
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return neuron_map

def generate_lists_for_exp(exp_type, base_dir, output_root):
    """
    exp_type: 'romanization' or 'shuffling'
    """
    configs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Define labels based on experiment type
    if exp_type == 'romanization':
        label_a = 'native'
        label_b = 'romanized'
        suffix_a = '.csv'
        suffix_b = '_romanized.csv'
    else:
        label_a = 'normal'
        label_b = 'shuffled'
        suffix_a = '.csv'
        suffix_b = '_shuffled.csv'

    for config in configs:
        is_raw = 'raw' in config
        config_path = os.path.join(base_dir, config)
        
        print(f"Processing {exp_type}/{config}...")
        
        neurons_a_map = get_neurons(config_path, suffix_a, is_raw)
        neurons_b_map = get_neurons(config_path, suffix_b, is_raw)
        
        all_langs = set(neurons_a_map.keys()) | set(neurons_b_map.keys())
        
        config_output_dir = os.path.join(output_root, exp_type, config)
        os.makedirs(config_output_dir, exist_ok=True)
        
        for lang in all_langs:
            dict_a = neurons_a_map.get(lang, {})
            dict_b = neurons_b_map.get(lang, {})
            
            set_a = set(dict_a.keys())
            set_b = set(dict_b.keys())
            
            overlap = set_a & set_b
            only_a = set_a - overlap
            only_b = set_b - overlap
            
            data = []
            for neuron in overlap:
                layer, idx = neuron
                data.append({
                    'layer': layer, 
                    'neuron_idx': idx, 
                    'category': 'overlap',
                    f'degree_{label_a}': dict_a[neuron],
                    f'degree_{label_b}': dict_b[neuron]
                })
            for neuron in only_a:
                layer, idx = neuron
                data.append({
                    'layer': layer, 
                    'neuron_idx': idx, 
                    'category': f'only_{label_a}',
                    f'degree_{label_a}': dict_a[neuron],
                    f'degree_{label_b}': None
                })
            for neuron in only_b:
                layer, idx = neuron
                data.append({
                    'layer': layer, 
                    'neuron_idx': idx, 
                    'category': f'only_{label_b}',
                    f'degree_{label_a}': None,
                    f'degree_{label_b}': dict_b[neuron]
                })
            
            if data:
                df = pd.DataFrame(data)
                # Reorder columns to be more readable
                cols = ['layer', 'neuron_idx', 'category', f'degree_{label_a}', f'degree_{label_b}']
                df = df[cols]
                df = df.sort_values(['layer', 'neuron_idx'])
                output_path = os.path.join(config_output_dir, f"{lang}.csv")
                df.to_csv(output_path, index=False)

if __name__ == "__main__":
    language_specific_features_root = "/home/user/language-specific-features"
    csv_root = os.path.join(language_specific_features_root, "csvs")
    output_root = os.path.join(language_specific_features_root, "neuron_lists")
    
    # Romanization
    generate_lists_for_exp('romanization', os.path.join(csv_root, 'romanization'), output_root)
    
    # Shuffling
    generate_lists_for_exp('shuffling', os.path.join(csv_root, 'shuffling'), output_root)
    
    print(f"\nDone! Lists generated in {output_root}")

