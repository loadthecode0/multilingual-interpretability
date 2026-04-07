import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import sys
from pathlib import Path

# Add parent directory to sys.path to find models, utils, etc.
sys.path.append(str(Path(__file__).resolve().parent.parent))

# project imports
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils import config, paths
from tqdm import tqdm
# -----------------------------
# CLI
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model (e.g. HuggingFace ID or local path)")
    parser.add_argument('--model_name', type=str, default='Llama-3.2-1B')
    parser.add_argument("--sae-model", type=str, required=True,
                        help="Trained SAE path")
    parser.add_argument("--method", type=str, default="sae_lape")
    parser.add_argument("--layers", nargs="+", required=True,
                        help="Layers to process, e.g. 0 1 2")
    parser.add_argument("--langs", nargs="+", required=True,
                        help="Languages, e.g. en de fr it")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment descriptor, e.g. jw300-thresh-50")
    parser.add_argument("--features", type=str, nargs="+", required=True,
                        help="Feature CSV names without prefix, e.g. geo fam")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-acts", action="store_true",
                        help="If set, save activations before probing")
    parser.add_argument("--all-neurons", action="store_true",
                        help="If set, run probing for all SAE neurons instead of just loaded ones")
    parser.add_argument("--use-shared", action="store_true",
                    help="If set, probe only neurons shared across multiple languages")
    parser.add_argument("--raw-model", action="store_true",
                    help="If set, probe raw hidden states instead of SAE latents")

    return parser.parse_args()

# -----------------------------
# Load identified neuron indices with source tracking
# -----------------------------
def load_neuron_indices_with_sources(exp, model_name, method, layers, langs, split):
    indices = {int(l): {} for l in layers}
    # Track which languages contribute each neuron
    neuron_sources = {int(l): defaultdict(list) for l in layers}
    # print(langs)
    
    base_dir = f"identification/{os.path.basename(model_name)}/{method}"
    for layer in layers:
        layer_int = int(layer)
        for lang in langs:
            csv_path = os.path.join(base_dir, f"layer_{layer}", exp, split, f"{lang}.csv")
            if os.path.exists(csv_path):
                print(lang)
                df = pd.read_csv(csv_path)
                neuron_list = df["feature_idx"].tolist()
                indices[layer_int][lang] = neuron_list
                # print(len(neuron_list))
                # Track source languages for each neuron
                for neuron_idx in neuron_list:
                    neuron_sources[layer_int][neuron_idx].append(lang)
            else:
                indices[layer_int][lang] = []
    
    return indices, neuron_sources

def load_shared_neuron_indices(exp, model_name, method, layers, split):
    indices = {int(l): [] for l in layers}
    neuron_sources = {int(l): {} for l in layers}

    base_dir = f"identification/{os.path.basename(model_name)}/{method}"
    for layer in layers:
        l = int(layer)
        csv_path = os.path.join(base_dir, f"layer_{layer}", exp, split, "shared_neurons.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing shared neuron file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        neuron_list = df["feature_idx"].tolist()
        indices[l] = neuron_list

        for row in df.itertuples():
            neuron_sources[l][row.feature_idx] = row.languages.split(",")

        print(f"Layer {l}: Loaded {len(neuron_list)} shared neurons "
              f"(avg num_languages={df['num_languages'].mean():.2f})")

    return indices, neuron_sources


# -----------------------------
# Load Lang2Vec features from CSV
# -----------------------------
def load_feature_csvs(langs, feature_names):
    feats = {}
    for fset in feature_names:
        path = os.path.join("lang2vec_probing/features", f"lang2vec_{fset}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_csv(path, index_col=0)
        df = df.loc[df.index.intersection(langs)]
        feats[fset] = df
    return feats

# -----------------------------
# Collect activations for selected neurons
# -----------------------------
def collect_activations(model, saes, dataset_manager, langs, layers, neuron_indices, 
                        neuron_sources, split, batch_size, device, 
                        all_neurons=False, use_shared=False, raw_model=False):
    model.to(device)
    model.eval()
    print(langs)

    lang_to_acts = {lang: {} for lang in langs}
    
    # Compute the union of neuron indices and track sources
    union_indices = {}
    union_sources = {}
    
    for layer in layers:
        l = int(layer)
        
        if all_neurons:
            # Use all neurons from SAE
            layer_name = f"layers.{l}.mlp"
            if raw_model:
                total_neurons = model.config.hidden_size
                union_indices[l] = list(range(total_neurons))
                union_sources[l] = {i: [] for i in range(total_neurons)}
                print(f"Layer {l}: Using all {total_neurons} model neurons")
            
            elif layer_name in saes:
                sae = saes[layer_name]
                if hasattr(sae,"num_latents"):
                    total_neurons = sae.num_latents  # Assuming d_sae contains the total number of neurons
                else:
                    total_neurons = sae.cfg.d_sae
                union_indices[l] = list(range(total_neurons))
                union_sources[l] = {i: [] for i in range(total_neurons)}  # no sources tracked
                print(f"Layer {l}: Using all {total_neurons} SAE neurons")
            else:
                print(f"[WARN] SAE not found for layer {l}, skipping")
                union_indices[l] = []
                union_sources[l] = {}

        elif use_shared and not raw_model:
            # Shared neurons already come as a flat list
            all_indices = set(neuron_indices[l])
            union_indices[l] = sorted(list(all_indices))
            
            # Source langs come directly from shared CSV
            union_sources[l] = {}
            for neuron_idx in union_indices[l]:
                union_sources[l][neuron_idx] = neuron_sources[l].get(neuron_idx, [])
            
            print(f"Layer {l}: Loaded {len(union_indices[l])} shared neurons")
            print(f"  Example: neuron {union_indices[l][0]} → {union_sources[l][union_indices[l][0]]}")

        else:
            # Selective neurons from per-language identification
            all_indices = set()
            for lang in langs:
                if neuron_indices[l][lang]:
                    all_indices.update(neuron_indices[l][lang])
            
            union_indices[l] = sorted(list(all_indices))
            
            union_sources[l] = {}
            for neuron_idx in union_indices[l]:
                union_sources[l][neuron_idx] = neuron_sources[l][neuron_idx]
            
            print(f"Layer {l}: Union of {len(union_indices[l])} unique neurons across languages")
            
            # Print distribution
            source_counts = {}
            for neuron_idx in union_indices[l]:
                num_sources = len(union_sources[l][neuron_idx])
                source_counts[num_sources] = source_counts.get(num_sources, 0) + 1
            
            print(f"  Neuron distribution by number of source languages:")
            for num_sources in sorted(source_counts.keys()):
                print(f"    {num_sources} language(s): {source_counts[num_sources]} neurons")

    # ---- Collect activations ----
    for lang in langs:
        dl = dataset_manager.create_dataloader(
            "flores_plus", lang, split,
            batch_size=batch_size, shuffle=False
        )
        if dl is None:
            print(f"[WARN] Dataloader for {lang} is None")
            continue
        
        if raw_model: 
            saes = {f"layers.{l}.mlp": None for l in layers}

        for layer_name, sae in saes.items():
            l = int(layer_name.split(".")[1])
            if not union_indices.get(l):
                print(f"No neurons in union for layer {l}")
                continue

            if not raw_model:
                sae = sae.to(device)
            collected = []

            print(f"Collecting activations for {lang} | Layer {l} | Union neurons: {len(union_indices[l])}")

            for batch in tqdm(dl, desc=f"{lang} | Layer {l}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True)
                    hidden = outputs.hidden_states[int(l) + 1]  # (B,T,H)
                    if not raw_model:
                        sae_out = sae.encode(hidden)
                        if hasattr(sae_out,'pre_acts'):
                            latents = sae_out.pre_acts  # (B,T,N)
                        else:
                            latents = sae_out
                    else:
                        latents = hidden
                
                    selected = latents[:, :, union_indices[l]]  # (B,T,K)
                    collected.append(selected.mean(dim=(0, 1)).cpu())
                    # else:
                    #     collected.append(hidden[:, :, union_indices[l]].mean(dim=(0, 1)).cpu())
            if collected:
                lang_to_acts[lang][l] = torch.stack(collected).mean(dim=0).float().numpy()
            
            if not raw_model:
                sae.to("cpu")
            torch.cuda.empty_cache()

    return lang_to_acts, union_indices, union_sources

# -----------------------------
# Run probes with source language tracking
# -----------------------------
# def run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
#                             langs, layers, out_dir, all_neurons=False, use_shared=False):
#     import os
#     import math
#     import numpy as np
#     import torch
#     from sklearn.model_selection import KFold
#     from tqdm import tqdm

#     os.makedirs(out_dir, exist_ok=True)
#     device = "cuda"

#     # Hyperparams (tweakable)
#     alpha = 1.0          # ridge regularization (λ)
#     n_splits = 5         # K in K-fold CV
#     feat_chunk = 256     # chunk size for feature-dim processing to limit memory use
#     WRITE_CHUNK = 2000000

#     for fset, feat_df in features.items():
#         print(fset)
#         results = []

#         for l in layers:
#             l = int(l)

#             # Only use languages that have activations for this layer
#             lang_subset = [lang for lang in langs if l in lang_to_acts[lang]]
#             if len(lang_subset) < 2:
#                 continue

#             print(f"\n[Layer {l}] Languages: {lang_subset}")

#             # Activation matrix: (L, K)
#             X = np.stack([lang_to_acts[lang][l] for lang in lang_subset], axis=0)
#             neuron_ids = union_indices[l]
#             feat_subdf_full = feat_df.loc[lang_subset]  # (L, F_orig)

#             print(f" - Activations: {X.shape}  (langs × neurons)")
#             print(f" - Features (before filtering):    {feat_subdf_full.shape}  (langs × feat_dims)")

#             # --- drop zero-variance feature columns (across the selected languages) ---
#             feat_var = feat_subdf_full.var(axis=0)
#             keep_mask = feat_var > 0
#             if keep_mask.sum() == 0:
#                 print(f"[WARN] Feature set {fset} has zero variance across selected languages for layer {l}; skipping.")
#                 continue

#             feat_subdf = feat_subdf_full.loc[:, keep_mask]
#             feat_names = list(feat_subdf.columns)
#             F = feat_subdf.shape[1]

#             print(f" - Features (after zero-var removal): {feat_subdf.shape} (kept {F})")

#             # Convert to torch tensors on device for heavy ops
#             X_t_full = torch.tensor(X, dtype=torch.float32, device=device)   # (L, K)
#             Y_full = torch.tensor(feat_subdf.values, dtype=torch.float32, device=device)  # (L, F)

#             L = X_t_full.shape[0]
#             K = X_t_full.shape[1]

#             # We'll accumulate fold-wise R2 for each neuron and feature dim on CPU to avoid GPU memory growth.
#             # r2_accum shape: (K, F) on CPU
#             r2_accum = np.zeros((K, F), dtype=np.float64)

#             # Prepare CV splits on language indices (deterministic)
#             kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#             splits = list(kf.split(np.arange(L)))

#             print(f"Running {n_splits}-fold CV: {len(splits)} splits")

#             # For each fold, compute per-neuron betas and evaluate on holdout.
#             for fold_idx, (train_idx, test_idx) in enumerate(splits):
#                 print(f" Fold {fold_idx+1}/{n_splits}  (train {len(train_idx)} | test {len(test_idx)})")

#                 # index into tensors
#                 X_tr = X_t_full[train_idx]   # (L_tr, K)
#                 X_te = X_t_full[test_idx]    # (L_te, K)

#                 Y_tr = Y_full[train_idx]     # (L_tr, F)
#                 Y_te = Y_full[test_idx]      # (L_te, F)

#                 # We'll perform per-neuron regressions independently.
#                 # To keep things efficient we operate in feature-dimension chunks.

#                 # Precompute denominators per neuron for ridge:
#                 # denom_k = sum(x_tr[:,k]^2) + alpha  -> shape (K,)
#                 denom = (X_tr ** 2).sum(dim=0).clamp(min=1e-12) + alpha   # (K,)

#                 # For each chunk of feature dims
#                 for start in range(0, F, feat_chunk):
#                     end = min(start + feat_chunk, F)
#                     fcount = end - start

#                     # slice Y (train/test) for chunk
#                     Y_tr_chunk = Y_tr[:, start:end]   # (L_tr, fcount)
#                     Y_te_chunk = Y_te[:, start:end]   # (L_te, fcount)

#                     # We'll compute numerator per neuron: (x_tr[:,k] dot Y_tr_chunk)  -> (K, fcount)
#                     # Compute with matmul: X_tr.T @ Y_tr_chunk  -> (K, fcount)
#                     # Note: X_tr is (L_tr, K)
#                     numer = X_tr.T @ Y_tr_chunk       # (K, fcount)

#                     # beta for each neuron and each feature-dim in chunk: (K, fcount)
#                     # denom is (K,) -> reshape to (K,1)
#                     beta_chunk = numer / denom.unsqueeze(1)   # (K, fcount)

#                     # Now compute predictions on test:
#                     # For neuron k: y_pred = x_te[:,k].unsqueeze(1) * beta_chunk[k]  -> sum over neurons?
#                     # IMPORTANT: This code intentionally computes **per-neuron single-predictor** predictions:
#                     # that is, each neuron individually predicts Y (1->F), so prediction by neuron k is:
#                     #    Y_pred_k = x_te[:,k:k+1] * beta_chunk[k:k+1, :]  -> (L_te, fcount)
#                     # We need per-neuron residual sums.
#                     # We'll compute for all neurons at once using broadcasting:
#                     # X_te: (L_te, K), beta_chunk: (K, fcount)
#                     # -> X_te.unsqueeze(2) * beta_chunk.unsqueeze(0) => (L_te, K, fcount)
#                     # But that can be large; instead compute per-neuron in a loop over neurons in vectorized blocks.

#                     # To balance memory/time, vectorize across neurons in blocks as well.
#                     neuron_block = 512  # tuneable: number of neurons to process in one inner block
#                     for nstart in range(0, K, neuron_block):
#                         nend = min(nstart + neuron_block, K)
#                         nb = nend - nstart

#                         # slice relevant tensors
#                         X_te_blk = X_te[:, nstart:nend]         # (L_te, nb)
#                         beta_blk = beta_chunk[nstart:nend, :]   # (nb, fcount)

#                         # predictions: Y_pred_blk per neuron block
#                         # compute elementwise product with broadcasting:
#                         # X_te_blk.unsqueeze(2): (L_te, nb, 1)
#                         # beta_blk.unsqueeze(0): (1, nb, fcount)
#                         # product => (L_te, nb, fcount)
#                         Y_pred_blk = (X_te_blk.unsqueeze(2) * beta_blk.unsqueeze(0))  # (L_te, nb, fcount)

#                         # compute residual sum of squares per neuron & feature: sum over test langs
#                         # ss_res_blk: (nb, fcount)
#                         ss_res_blk = ((Y_te_chunk.unsqueeze(1) - Y_pred_blk) ** 2).sum(dim=0)  # sum over L_te -> (nb, fcount)

#                         # compute ss_tot per neuron & feature: variance relative to train mean
#                         # Need train mean for this feature-chunk:
#                         Y_tr_mean_chunk = Y_tr_chunk.mean(dim=0, keepdim=True)  # (1, fcount)
#                         ss_tot_blk = ((Y_te_chunk.unsqueeze(1) - Y_tr_mean_chunk.unsqueeze(1)) ** 2).sum(dim=0)  # (nb, fcount)

#                         # r2 per neuron-feature for this block and chunk
#                         # avoid division by zero
#                         denom_ss = ss_tot_blk.clone()
#                         denom_ss[denom_ss == 0] = float("nan")  # will produce nan R2 for degenerate cases
#                         r2_blk = 1.0 - (ss_res_blk / denom_ss)   # (nb, fcount)

#                         # move to CPU and accumulate into r2_accum
#                         r2_accum[nstart:nend, start:end] += r2_blk.detach().cpu().numpy()

#                         # free GPU mem
#                         del X_te_blk, beta_blk, Y_pred_blk, ss_res_blk, ss_tot_blk, r2_blk
#                         torch.cuda.empty_cache()

#                     # end neuron_block loop

#                 # end feature chunk loop

#                 # free fold-level tensors to reduce GPU memory
#                 del X_tr, X_te, Y_tr, Y_te
#                 torch.cuda.empty_cache()

#             # end fold loop

#             # Average across folds
#             r2_mean = r2_accum / float(n_splits)   # (K, F)

#             # --- Streaming write to CSV (same filename logic as original) ---
#             print("Streaming write to CSV (no in-memory results)")

#             if all_neurons:
#                 filename = f"{fset}_probes_all_neurons.csv"
#             elif use_shared:
#                 filename = f"{fset}_probes_shared.csv"
#             else:
#                 filename = f"{fset}_probes_with_sources.csv"

#             out_path = os.path.join(out_dir, filename)

#             # Write header once
#             if not os.path.exists(out_path):
#                 with open(out_path, "w") as f:
#                     f.write("layer,neuron_idx,source_languages,num_source_langs,feature_set,feature_name,feature_idx,r2_score\n")

#             buffer = []
#             written = 0

#             for ni, neuron_id in tqdm(enumerate(neuron_ids), total=len(neuron_ids), desc="Writing results"):
#                 source_langs = union_sources[l].get(neuron_id, [])
#                 num_src = len(source_langs)
#                 src_str = ",".join(sorted(source_langs))

#                 row = r2_mean[ni]  # (F,)

#                 for fi, feat_name in enumerate(feat_names):
#                     score = float(row[fi])
#                     if math.isnan(score):
#                         continue

#                     buffer.append(f"{l},{neuron_id},{src_str},{num_src},{fset},{feat_name},{fi},{score}\n")

#                     if len(buffer) >= WRITE_CHUNK:
#                         with open(out_path, "a") as f:
#                             f.writelines(buffer)
#                         written += len(buffer)
#                         buffer = []
#                         print(f"  [FLUSH] wrote {written:,} rows so far")

#             # final flush
#             if buffer:
#                 with open(out_path, "a") as f:
#                     f.writelines(buffer)
#                 written += len(buffer)

#             print(f" → Finished layer {l}, total rows written: {written:,}")

#     # end outermost loops

def run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            langs, layers, out_dir, all_neurons=False, use_shared=False):
    import os
    import math
    import numpy as np
    import torch
    from sklearn.model_selection import KFold
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"

    # Hyperparams
    alpha = 1.0          # ridge regularization (λ)
    n_splits = 5         # K in K-fold CV
    feat_chunk = 256     # chunk size for feature-dim processing to limit memory use
    neuron_block = 512   # block size for neurons
    WRITE_CHUNK = 2000000

    for fset, feat_df in features.items():
        print(fset)

        for l in layers:
            l = int(l)

            # Only use languages that have activations for this layer
            lang_subset = [lang for lang in langs if l in lang_to_acts[lang]]
            if len(lang_subset) < 2:
                continue

            print(f"\n[Layer {l}] Languages: {lang_subset}")

            # Activation matrix: (L, K)
            X = np.stack([lang_to_acts[lang][l] for lang in lang_subset], axis=0)
            neuron_ids = union_indices[l]
            feat_subdf_full = feat_df.loc[lang_subset]  # (L, F_orig)

            # --- drop zero-variance feature columns ---
            print(f" - Features (before zero-var removal): {feat_subdf_full.shape}")

            feat_var = feat_subdf_full.var(axis=0)
            keep_mask = feat_var > 1e-12
            if keep_mask.sum() == 0:
                print(f"[WARN] Feature set {fset} has zero variance across selected languages for layer {l}; skipping.")
                continue
            feat_subdf = feat_subdf_full.loc[:, keep_mask]
            feat_names = list(feat_subdf.columns)
            F = feat_subdf.shape[1]

            print(f" - Activations: {X.shape}  (langs × neurons)")
            print(f" - Features (after zero-var removal): {feat_subdf.shape}")

            # Convert to torch tensors
            X_t_full = torch.tensor(X, dtype=torch.float32, device=device)       # (L, K)
            Y_full = torch.tensor(feat_subdf.values, dtype=torch.float32, device=device)  # (L, F)
            L, K = X_t_full.shape

            # Accumulate R² across folds
            r2_accum = np.zeros((K, F), dtype=np.float64)

            # K-fold CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(np.arange(L)))

            print(f"Running {n_splits}-fold CV")

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_tr = X_t_full[train_idx]   # (L_tr, K)
                X_te = X_t_full[test_idx]    # (L_te, K)
                Y_tr = Y_full[train_idx]     # (L_tr, F)
                Y_te = Y_full[test_idx]      # (L_te, F)

                # Process feature chunks
                for start in range(0, F, feat_chunk):
                    end = min(start + feat_chunk, F)
                    fcount = end - start

                    Y_tr_chunk = Y_tr[:, start:end]
                    Y_te_chunk = Y_te[:, start:end]

                    # center Y
                    Y_tr_mu = Y_tr_chunk.mean(dim=0, keepdim=True)
                    Y_te_mu = Y_tr_mu
                    Y_tr_c = Y_tr_chunk - Y_tr_mu
                    Y_te_c = Y_te_chunk - Y_te_mu

                    # ridge β: per-neuron
                    denom = (X_tr ** 2).sum(dim=0) + alpha   # (K,)
                    numer = X_tr.T @ Y_tr_c                  # (K, fcount)
                    beta_chunk = numer / denom.unsqueeze(1)  # (K, fcount)

                    # process neuron blocks
                    for nstart in range(0, K, neuron_block):
                        nend = min(nstart + neuron_block, K)
                        nb = nend - nstart

                        X_te_blk = X_te[:, nstart:nend]
                        beta_blk = beta_chunk[nstart:nend, :]

                        # prediction
                        Y_pred_c_blk = X_te_blk.unsqueeze(2) * beta_blk.unsqueeze(0)  # (L_te, nb, fcount)
                        Y_pred_blk = Y_pred_c_blk + Y_te_mu.unsqueeze(1)

                        # residual sum of squares
                        ss_res_blk = ((Y_te_chunk.unsqueeze(1) - Y_pred_blk) ** 2).sum(dim=0)

                        # total sum of squares
                        ss_tot_blk = ((Y_te_chunk.unsqueeze(1) - Y_tr_mu.unsqueeze(1)) ** 2).sum(dim=0)
                        ss_tot_blk[ss_tot_blk < 1e-12] = float("nan")

                        r2_blk = 1.0 - (ss_res_blk / ss_tot_blk)
                        r2_accum[nstart:nend, start:end] += r2_blk.detach().cpu().numpy()

                        del X_te_blk, beta_blk, Y_pred_blk, Y_pred_c_blk, ss_res_blk, ss_tot_blk, r2_blk
                        torch.cuda.empty_cache()

                del X_tr, X_te, Y_tr, Y_te
                torch.cuda.empty_cache()

            # average across folds
            r2_mean = r2_accum / float(n_splits)

            # --- streaming CSV write ---
            if all_neurons:
                filename = f"{fset}_probes_all_neurons.csv"
            elif use_shared:
                filename = f"{fset}_probes_shared.csv"
            else:
                filename = f"{fset}_probes_with_sources.csv"

            out_path = os.path.join(out_dir, filename)
            if not os.path.exists(out_path):
                with open(out_path, "w") as f:
                    f.write("layer,neuron_idx,source_languages,num_source_langs,feature_set,feature_name,feature_idx,r2_score\n")

            buffer = []
            written = 0

            for ni, neuron_id in tqdm(enumerate(neuron_ids), total=len(neuron_ids), desc="Writing results"):
                source_langs = union_sources[l].get(neuron_id, [])
                num_src = len(source_langs)
                src_str = ",".join(sorted(source_langs))

                row = r2_mean[ni]

                for fi, feat_name in enumerate(feat_names):
                    score = float(row[fi])
                    if math.isnan(score):
                        continue
                    buffer.append(f"{l},{neuron_id},{src_str},{num_src},{fset},{feat_name},{fi},{score}\n")

                    if len(buffer) >= WRITE_CHUNK:
                        with open(out_path, "a") as f:
                            f.writelines(buffer)
                        written += len(buffer)
                        buffer = []
                        print(f"  [FLUSH] wrote {written:,} rows so far")

            if buffer:
                with open(out_path, "a") as f:
                    f.writelines(buffer)
                written += len(buffer)

            print(f" → Finished layer {l}, total rows written: {written:,}")


# -----------------------------
# Main function
# -----------------------------
def main():
    args = get_args()
    print(args.features)
    args.layer_names = args.layers
    args.layers = [layer.split(".")[1] for layer in args.layers]
    logger = config.get_logger()

    # Load model + SAE
    model_loader = m_loader.HFModelLoader(args.model_path, "llm", args.device, logger)
    model = model_loader.model

    if not args.raw_model:
        sae_loader = m_loader.SAELoader(args.sae_model, args.layer_names, args.device, logger)
        saes = sae_loader.sae_model

    dataset_manager = MultilingualDatasetManager(model_name=args.model_path)

    # Load neuron indices with source tracking (only needed if not using all neurons)
    if args.raw_model:
        print("[INFO] Running probing on raw model activations (no SAE)")
        saes = {}              # do not load SAE
        neuron_indices = {int(l): {lang: list(range(model.config.hidden_size))
                                for lang in args.langs}
                        for l in args.layers}
        neuron_sources = {int(l): {i: [] for i in range(model.config.hidden_size)}
                        for l in args.layers}
    elif args.all_neurons:
        print("[INFO] Using all SAE neurons for probing")
        neuron_indices, neuron_sources = {}, {}
    elif args.use_shared:
        print("[INFO] Using only shared neurons across languages")
        neuron_indices, neuron_sources = load_shared_neuron_indices(
            args.exp, args.model_name, args.method, args.layers, args.split
        )
    else:
        print("[INFO] Using selective neurons from identification results")
        neuron_indices, neuron_sources = load_neuron_indices_with_sources(
            args.exp, args.model_name, args.method, args.layers, args.langs, args.split
        )

    # print(neuron_indices)

    # Collect activations for union of neurons
    lang_to_acts, union_indices, union_sources = collect_activations(
        model, saes, dataset_manager,
        args.langs, args.layers, neuron_indices, neuron_sources,
        args.split, args.batch_size, args.device, args.all_neurons, args.use_shared,
        args.raw_model
    )

    # Optional save
    if args.save_acts:
        out_act_dir = f"lang2vec_probing/results/{args.exp}/activations"
        os.makedirs(out_act_dir, exist_ok=True)
        torch.save(lang_to_acts, os.path.join(out_act_dir, "activations.pt"))
        torch.save(union_indices, os.path.join(out_act_dir, "union_indices.pt"))
        torch.save(union_sources, os.path.join(out_act_dir, "union_sources.pt"))
        print(f"[INFO] Saved activations, union indices, and sources → {out_act_dir}")

    # Load lang2vec features
    print(args.features)
    features = load_feature_csvs(args.langs, args.features)

    # Run probes with source tracking
    if "llama" in args.model_path.lower():
        if args.raw_model:
            out_dir = f"lang2vec_probing/results_raw_cv/{args.layers[0]}/{args.exp}"
        else:
            out_dir = f"lang2vec_probing/results_cv/{args.layers[0]}/{args.exp}"
    else:
        if args.raw_model:
            out_dir = f"lang2vec_probing/gemma_results_raw_cv/{args.layers[0]}/{args.exp}"
        else:
            out_dir = f"lang2vec_probing/gemma_results_cv/{args.layers[0]}/{args.exp}"
    
    run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            args.langs, args.layers, out_dir, args.all_neurons, args.use_shared)

if __name__ == "__main__":
    main()