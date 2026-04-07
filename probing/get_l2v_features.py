import argparse
import json
import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path to find utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import paths

import lang2vec.lang2vec as l2v
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

def load_union_of_langs(config_path):
    """Collect union of supported_languages across all datasets."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    langs = set()
    for ds in cfg["datasets"].values():
        langs.update(ds.get("supported_languages", []))
    return sorted(langs)

def normalize_lang_code(code):
    """Map dataset codes (eng_Latn, deu_Latn, etc.) to ISO codes used by lang2vec."""
    mapping = {
        "eng": "en", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "por": "pt",
        "rus": "ru", "zho": "zh", "jpn": "ja", "kor": "ko", "arb": "ar",
        "hin": "hi", "ben": "bn", "urd": "ur", "tur": "tr"
    }
    short = code.split("_")[0] if "_" in code else code
    return mapping.get(short, short)

def clean_dataframe_for_analysis(df):
    """Clean DataFrame by replacing '--' with NaN and handling missing values."""
    print(f"[INFO] Original shape: {df.shape}")
    print(f"[INFO] Missing values ('--') found: {(df == '--').sum().sum()}")
    
    # Replace '--' with NaN
    df_clean = df.replace('--', np.nan)
    
    # Convert to numeric, coercing errors to NaN
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    
    # Count NaN values after conversion
    nan_count = df_clean.isna().sum().sum()
    print(f"[INFO] NaN values after conversion: {nan_count}")
    
    # Fill missing values with column mean
    df_clean = df_clean.fillna(df_clean.mean())
    
    # Final check - fill any remaining NaN with 0 (in case entire columns were NaN)
    df_clean = df_clean.fillna(0)
    
    print(f"[INFO] Cleaned shape: {df_clean.shape}")
    return df_clean

def plot_pca(df, out_path, feature_set_name):
    """Create PCA plot with proper error handling."""
    df_clean = clean_dataframe_for_analysis(df)
    
    if len(df_clean) < 2:
        print(f"[WARNING] Not enough data points for PCA after cleaning: {len(df_clean)}")
        return
    
    if df_clean.shape[1] < 2:
        print(f"[WARNING] Not enough features for PCA: {df_clean.shape[1]}")
        return
        
    try:
        pca = PCA(n_components=min(2, df_clean.shape[1]))
        coords = pca.fit_transform(df_clean.values)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], s=60)
        
        # Add language labels
        for i, lang in enumerate(df_clean.index):
            plt.annotate(lang, (coords[i, 0], coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.title(f"Lang2Vec PCA projection - {feature_set_name}")
        plt.xlabel(f"PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (explained variance: {pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0:.2%})")
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved PCA plot → {out_path}")
        
    except Exception as e:
        print(f"[ERROR] PCA plotting failed: {e}")

def plot_tsne(df, out_path, feature_set_name):
    """Create t-SNE plot with proper error handling."""
    df_clean = clean_dataframe_for_analysis(df)
    
    if len(df_clean) < 2:
        print(f"[WARNING] Not enough data points for t-SNE after cleaning: {len(df_clean)}")
        return
        
    try:
        perplexity = min(30, max(1, len(df_clean) - 1))
        if perplexity < 5:
            perplexity = max(1, len(df_clean) // 3)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(df_clean.values)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], s=60)
        
        # Add language labels
        for i, lang in enumerate(df_clean.index):
            plt.annotate(lang, (coords[i, 0], coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.title(f"Lang2Vec t-SNE projection - {feature_set_name}")
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved t-SNE plot → {out_path}")
        
    except Exception as e:
        print(f"[ERROR] t-SNE plotting failed: {e}")

def plot_dendrogram(df, out_path, feature_set_name):
    """Create dendrogram with proper error handling."""
    df_clean = clean_dataframe_for_analysis(df)
    
    if len(df_clean) < 2:
        print(f"[WARNING] Not enough data points for clustering after cleaning: {len(df_clean)}")
        return
        
    try:
        Z = linkage(df_clean.values, method="ward")
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=df_clean.index.tolist(), leaf_rotation=90)
        plt.title(f"Lang2Vec hierarchical clustering - {feature_set_name}")
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved dendrogram → {out_path}")
        
    except Exception as e:
        print(f"[ERROR] Dendrogram plotting failed: {e}")

def monkey_patch_numpy_for_lang2vec():
    """Patch numpy.load to allow pickle loading for lang2vec compatibility."""
    original_load = np.load
    def patched_load(*args, **kwargs):
        if 'allow_pickle' not in kwargs:
            kwargs['allow_pickle'] = True
        return original_load(*args, **kwargs)
    np.load = patched_load
    return original_load

def restore_numpy_load(original_load):
    """Restore original numpy.load function."""
    np.load = original_load

def main(args):
    print("[INFO] Available Lang2Vec feature sets:")
    print(l2v.FEATURE_SETS)
    print()

    # Load and normalize language codes
    langs = load_union_of_langs(args.config_path)
    iso_langs = [normalize_lang_code(l) for l in langs]
    print(f"[INFO] Processing {len(iso_langs)} languages: {iso_langs}")
    print()

    # Patch numpy for lang2vec compatibility
    original_load = monkey_patch_numpy_for_lang2vec()

    base_path = os.path.dirname(args.out_csv)
    os.makedirs(base_path, exist_ok=True)
    
    # Create visualization directories
    vis_base = os.path.join(base_path, "l2v_feature_vis")
    os.makedirs(vis_base, exist_ok=True)
    
    # Process each feature set separately
    successful_feature_sets = []
    
    for feature_set in args.feature_set:
        try:
            print(f"[INFO] Processing feature set: {feature_set}")
            
            # Get features for this set
            features_dict = l2v.get_features(iso_langs, feature_set)
            df = pd.DataFrame.from_dict(features_dict, orient="index")
            
            if df.empty:
                print(f"[WARNING] No data returned for {feature_set}")
                continue
            
            print(f"[INFO] Raw features shape for {feature_set}: {df.shape}")
            
            # Clean the data
            df_clean = clean_dataframe_for_analysis(df)
            
            if df_clean.empty:
                print(f"[WARNING] No data remaining after cleaning for {feature_set}")
                continue
            
            # Add meaningful column names
            if feature_set == 'fam':
                # Family features - try to make them more interpretable
                df_clean.columns = [f"family_{i}" for i in range(len(df_clean.columns))]
            elif feature_set == 'geo':
                # Geographic features - often latitude/longitude or regional indicators
                df_clean.columns = [f"geo_{i}" for i in range(len(df_clean.columns))]
            elif feature_set == 'id':
                # Identity features
                df_clean.columns = [f"identity_{i}" for i in range(len(df_clean.columns))]
            else:
                # For other feature sets, use descriptive prefixes
                df_clean.columns = [f"{feature_set}_{i}" for i in range(len(df_clean.columns))]
            
            # Save individual CSV for this feature set
            out_file = os.path.join(base_path, f"lang2vec_{feature_set}.csv")
            df_clean.to_csv(out_file)
            print(f"[INFO] Saved {feature_set} features ({df_clean.shape}) → {out_file}")
            
            # Create visualizations for interpretable feature sets
            if feature_set in ['fam', 'geo', 'syntax_wals', 'phonology_wals', 'id'] and df_clean.shape[0] > 1:
                print(f"[INFO] Creating visualizations for {feature_set}...")
                
                vis_prefix = os.path.join(vis_base, feature_set)
                plot_pca(df, f"{vis_prefix}_pca.png", feature_set)
                plot_tsne(df, f"{vis_prefix}_tsne.png", feature_set)
                plot_dendrogram(df, f"{vis_prefix}_dendrogram.png", feature_set)
            else:
                vis_prefix = os.path.join(vis_base, feature_set)
                plot_pca(df, f"{vis_prefix}_pca.png", feature_set)
                # print(f"[INFO] Skipping visualizations for {feature_set} (too many dimensions or not interpretable)")
            
            successful_feature_sets.append((feature_set, df_clean))
            print(f"[INFO] Successfully processed {feature_set}")
            print("-" * 50)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {feature_set}: {e}")
            continue
    
    # Create combined CSV with all successful feature sets
    if successful_feature_sets:
        print(f"[INFO] Creating combined CSV with {len(successful_feature_sets)} feature sets...")
        
        combined_dfs = []
        for feature_set_name, df in successful_feature_sets:
            combined_dfs.append(df)
        
        # Concatenate all feature sets
        combined_df = pd.concat(combined_dfs, axis=1)
        
        # Save combined CSV
        combined_out = args.out_csv
        combined_df.to_csv(combined_out)
        print(f"[INFO] Saved combined features ({combined_df.shape}) → {combined_out}")
        
        # Create visualization for combined features if reasonable size
        if combined_df.shape[1] <= 100000:  # Only visualize if not too many dimensions
            print("[INFO] Creating visualizations for combined features...")
            vis_prefix = os.path.join(vis_base, "combined")
            plot_pca(combined_df, f"{vis_prefix}_pca.png", "Combined Features")
            plot_tsne(combined_df, f"{vis_prefix}_tsne.png", "Combined Features")
            plot_dendrogram(combined_df, f"{vis_prefix}_dendrogram.png", "Combined Features")
        else:
            print(f"[INFO] Skipping combined visualizations (too many dimensions: {combined_df.shape[1]})")
    
    else:
        print("[ERROR] No feature sets were successfully processed!")
    
    # Restore original numpy.load
    restore_numpy_load(original_load)
    
    print(f"\n[INFO] Processing complete! Check {base_path} for outputs:")
    print(f"  - Individual CSV files: lang2vec_<feature_set>.csv")
    print(f"  - Combined CSV: {os.path.basename(args.out_csv)}")
    print(f"  - Visualizations: {vis_base}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Lang2Vec features with proper handling of missing values")
    p.add_argument("--config_path", type=str, 
                   default=os.path.join(paths.PROJECT_ROOT, "data/multilingual_datasets/data_config.json"),
                   help="Path to dataset configuration JSON file")
    p.add_argument("--feature_set", type=str, nargs='+',
                   default=['syntax_wals', 'phonology_wals', 'syntax_sswl', 
                   'syntax_ethnologue', 'phonology_ethnologue', 'inventory_ethnologue', 
                   'inventory_phoible_aa', 'inventory_phoible_gm', 
                   'inventory_phoible_saphon', 'inventory_phoible_spa', 
                   'inventory_phoible_ph', 'inventory_phoible_ra', 
                   'inventory_phoible_upsid', 'syntax_knn', 'phonology_knn', 
                   'inventory_knn', 'syntax_average', 'phonology_average', 
                   'inventory_average', 'fam', 'id', 'geo'],
                   choices=['syntax_wals', 'phonology_wals', 'syntax_sswl', 
                   'syntax_ethnologue', 'phonology_ethnologue', 'inventory_ethnologue', 
                   'inventory_phoible_aa', 'inventory_phoible_gm', 
                   'inventory_phoible_saphon', 'inventory_phoible_spa', 
                   'inventory_phoible_ph', 'inventory_phoible_ra', 
                   'inventory_phoible_upsid', 'syntax_knn', 'phonology_knn', 
                   'inventory_knn', 'syntax_average', 'phonology_average', 
                   'inventory_average', 'fam', 'id', 'geo', 'learned'],
                   help="Feature sets to extract (can specify multiple)")
    p.add_argument("--out_csv", type=str, 
                   default=os.path.join(paths.PROJECT_ROOT, "probing/features/lang2vec_combined.csv"),
                   help="Path for combined output CSV file")
    
    args = p.parse_args()
    main(args)