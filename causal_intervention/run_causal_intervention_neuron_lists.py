#!/usr/bin/env python3
"""
Causal intervention script using pre-generated neuron lists.
Supports Romanization and Shuffling experiments.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path

# Add parent directory to sys.path to find models, utils, etc.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import loader as m_loader
from utils import config, paths
from data.multiloader import MultilingualDatasetManager


# -------------------------------------------------
# CLI
# -------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name", required=True, help="e.g. llama or gemma")
    p.add_argument("--sae-model", default=None)
    p.add_argument("--layer", nargs="+", default=None, help="Layers to intervene on (e.g. 8 9 or layers.8 layers.9). If not specified, all layers are used.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--raw-model", action="store_true")

    # neuron list config
    p.add_argument("--neuron-list-root", default=paths.NEURON_LISTS_DIR, help="Root directory for neuron lists")
    p.add_argument("--experiment-type", choices=["romanization", "shuffling"], required=True)
    p.add_argument("--romanization-mode", choices=["roman_dia", "roman_no_dia"], default=None)
    p.add_argument("--category", required=True, help="e.g. only_native, overlap, only_normal, etc.")
    p.add_argument("--max-degree", type=int, default=None, help="Filter neurons by degree of sharing (at most this value)")
    
    p.add_argument("--num-random-neurons", type=int, default=None, help="If provided, sample this many neurons randomly per layer instead of using the list")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")

    # ablation
    p.add_argument("--ablation", choices=["zero", "mean", "null"], required=True)
    p.add_argument("--mean-source", choices=["same", "other"], default="same")
    p.add_argument("--mean-lang", default=None)

    # data
    p.add_argument("--lang", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--max-examples", type=int, default=20)

    # output
    p.add_argument("--output-root", default="causal_results_neuron_lists")
    p.add_argument("--gen-tokens", type=int, default=20)

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Tunable argument for cross-ablation: h + alpha(mu_other - mu_curr)"
    )

    return p.parse_args()


def sample_random_neuron_ids(num_to_sample, model, model_name, raw_model, sae, seed):
    """Samples random neurons from the total pool."""
    if sae is not None:
        # Get dimension from SAE
        if hasattr(sae, "W_enc"):
            total_pool = sae.W_enc.shape[-1]
        elif hasattr(sae, "cfg"):
            total_pool = sae.cfg.d_sae
        else:
            total_pool = 131072 # Fallback for Llama-1B SAE
    else:
        # Get intermediate dimension for raw models
        if hasattr(model, "config") and hasattr(model.config, "intermediate_size"):
            total_pool = model.config.intermediate_size
        else:
            # Fallback based on model name
            if "llama" in model_name.lower():
                total_pool = 8192 # Intermediate size for Llama-3.2-1B
            elif "gemma" in model_name.lower():
                total_pool = 9216 # Intermediate size for Gemma-2-2b
            else:
                total_pool = 2048 # Fallback
            
    rng = np.random.default_rng(seed)
    return rng.choice(total_pool, size=num_to_sample, replace=False).tolist()


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_neurons_from_list(args, layer_idx):
    # Mapping for language codes to full names used in CSV filenames
    lang_map = {
        "bn": "Bengali",
        "bg": "Bulgarian",
        "zh": "Chinese",
        "en": "English",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "mr": "Marathi",
        "ru": "Russian",
        "es": "Spanish",
        "ur": "Urdu",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "th": "Thai",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "pt": "Portuguese",
    }
    # Use full name if code is provided, otherwise fallback to the string itself
    full_lang = lang_map.get(args.lang, args.lang)

    # Determine the directory name for the neuron lists
    rep_mode = "raw" if args.raw_model else "sae"
    
    # Normalize model name for lookup (llama or gemma)
    model_lookup = "llama" if "llama" in args.model_name.lower() else "gemma"
    
    if args.experiment_type == "romanization":
        mode_suffix = "_diacritics" if args.romanization_mode == "roman_dia" else "_no_diacritics"
        config_dir = f"{model_lookup}_{rep_mode}{mode_suffix}"
        label_a, label_b = "native", "romanized"
    else:
        config_dir = f"{model_lookup}_{rep_mode}"
        label_a, label_b = "normal", "shuffled"
        
    csv_path = os.path.join(
        args.neuron_list_root, args.experiment_type, config_dir, f"{full_lang}.csv"
    )
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Neuron list not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Filter by layer
    df = df[df["layer"] == layer_idx]
    
    # Filter by category
    df = df[df["category"] == args.category]
    
    # Filter by degree if specified
    if args.max_degree is not None:
        deg_col_a = f"degree_{label_a}"
        deg_col_b = f"degree_{label_b}"
        
        def get_max_deg(row):
            degs = []
            if pd.notna(row[deg_col_a]): degs.append(row[deg_col_a])
            if pd.notna(row[deg_col_b]): degs.append(row[deg_col_b])
            return max(degs) if degs else 0
            
        df["max_deg_val"] = df.apply(get_max_deg, axis=1)
        df = df[df["max_deg_val"] <= args.max_degree]
        
    neuron_ids = df["neuron_idx"].tolist()
    print(f"Loaded {len(neuron_ids)} neurons from {csv_path} for category {args.category}")
    if neuron_ids:
        print(f"[DEBUG] First 10 neuron indices: {neuron_ids[:10]}")
        print(f"[DEBUG] Max neuron index: {max(neuron_ids)}")
    return neuron_ids


def kl_divergence(logits_clean, logits_patch):
    p = torch.log_softmax(logits_clean, dim=-1)
    q = torch.log_softmax(logits_patch, dim=-1)
    return torch.sum(torch.exp(p) * (p - q), dim=-1)


def compute_ppl(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
    return float(torch.exp(out.loss))


# -------------------------------------------------
# Mean activations
# -------------------------------------------------
def compute_mean_vectors(
    model, dataloader, layer_configs, device, raw_model, max_batches=100
):
    """
    Computes mean vectors for multiple layers in a single pass.
    layer_configs: list of dicts { 'idx': int, 'module': nn.Module, 'sae': sae_model or None }
    Returns: dict of { layer_idx: mean_vector }
    """
    acc = {cfg['idx']: [] for cfg in layer_configs}
    # To store captured activations per batch
    batch_acts = {cfg['idx']: None for cfg in layer_configs}

    def make_hook(l_idx, sae):
        def hook_fn(module, inp, out):
            if raw_model:
                batch_acts[l_idx] = out.detach().cpu()
            else:
                sae_out = sae.encode(out)
                batch_acts[l_idx] = sae_out.pre_acts.detach().cpu()
        return hook_fn

    handles = []
    for cfg in layer_configs:
        h = cfg['module'].register_forward_hook(make_hook(cfg['idx'], cfg.get('sae')))
        handles.append(h)

    try:
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            for l_idx in acc:
                if batch_acts[l_idx] is not None:
                    # [B, T, D] -> mean over B, T -> [D]
                    acc[l_idx].append(batch_acts[l_idx].mean(dim=(0, 1)))
                    batch_acts[l_idx] = None
    finally:
        for h in handles:
            h.remove()

    results = {}
    for l_idx, vectors in acc.items():
        if not vectors:
            print(f"[WARNING] No activations captured for layer {l_idx}")
            continue
        results[l_idx] = torch.stack(vectors).mean(dim=0)

    return results


# -------------------------------------------------
# Hook for causal intervention
# -------------------------------------------------
def make_patch_hook(
    neuron_ids,
    mu_other,
    mu_curr,
    ablation,
    raw_model,
    sae,
    active_dict,
    alpha=1.0,
    mean_source="same",
):
    neuron_ids = list(neuron_ids)

    def hook(module, inp, out):
        if not active_dict["active"]:
            return out
        
        if ablation == "null":
            return out

        device = out.device
        dtype = out.dtype
        
        # [DEBUG] Check dimensions
        if raw_model:
            if not hasattr(hook, "_debug_done"):
                print(f"[DEBUG HOOK] out.shape: {out.shape}")
                print(f"[DEBUG HOOK] mu_other.shape: {mu_other.shape if mu_other is not None else 'None'}")
                print(f"[DEBUG HOOK] max(neuron_ids): {max(neuron_ids) if neuron_ids else 'N/A'}")
                hook._debug_done = True
        
        # ======================================================
        # RAW MODEL
        # ======================================================
        if raw_model:
            h = out.clone()
            
            # Check for index out of bounds
            valid_neuron_ids = [nid for nid in neuron_ids if nid < h.shape[-1]]
            if len(valid_neuron_ids) < len(neuron_ids):
                if not hasattr(hook, "_warn_done"):
                    print(f"[WARNING] Some neuron indices are out of bounds for tensor of shape {h.shape}. "
                          f"Using {len(valid_neuron_ids)}/{len(neuron_ids)} valid neurons.")
                    hook._warn_done = True
            
            if not valid_neuron_ids:
                return out

            idx = torch.tensor(valid_neuron_ids, device=device)
            old_vals = out[:, :, idx].clone()

            if ablation == "zero":
                h[:, :, idx] = 0.0
            else:
                if mean_source == "other":
                    diff = (mu_other[idx] - mu_curr[idx]).to(device=device, dtype=h.dtype)
                    h[:, :, idx] = h[:, :, idx] + alpha * diff
                else:
                    h[:, :, idx] = mu_other[idx].to(device=device, dtype=h.dtype)
            
            new_vals = h[:, :, idx]
            diff_vals = (new_vals - old_vals).abs()
            num_changed = (diff_vals > 1e-5).sum().item()
            avg_increase = (new_vals - old_vals).mean().item()
            print(f"[DEBUG RAW] Neurons changed: {num_changed} | Avg value increase: {avg_increase:.6f}")

            return h.to(dtype)

        # ======================================================
        # SPARSECODER (Top-K)
        # ======================================================
        enc = sae.encode(out)
        top_acts = enc.top_acts.clone()
        top_indices = enc.top_indices.clone()
        B, T, K = top_acts.shape

        total_bumped_out = 0
        total_mass_increase = 0.0

        for b in range(B):
            for t in range(T):
                acts_bt = top_acts[b, t]
                idxs_bt = top_indices[b, t]
                candidate_vals = acts_bt.tolist()
                candidate_idxs = idxs_bt.tolist()
                original_topk_set = set(candidate_idxs)
                original_total_mass = sum(candidate_vals)
                original_topk_vals = {idx: val for idx, val in zip(candidate_idxs, candidate_vals)}

                for nid in neuron_ids:
                    h_nid = original_topk_vals.get(nid, 0.0)
                    if ablation == "zero":
                        h_prime = 0.0
                    elif ablation == "mean":
                        if mean_source == "other":
                            v_other = mu_other[nid].item()
                            v_curr = mu_curr[nid].item()
                            h_prime = h_nid + alpha * (v_other - v_curr)
                        else:
                            h_prime = mu_other[nid].item()
                        h_prime = max(0.0, h_prime)
                    else:
                        h_prime = h_nid

                    if nid in original_topk_set:
                        for k_idx in range(K):
                            if candidate_idxs[k_idx] == nid:
                                candidate_vals[k_idx] = h_prime
                                break
                    else:
                        candidate_vals.append(h_prime)
                        candidate_idxs.append(nid)

                combined = sorted(zip(candidate_vals, candidate_idxs), key=lambda x: x[0], reverse=True)
                new_top_vals, new_top_idxs = zip(*combined[:K])

                new_topk_set = set(new_top_idxs)
                new_total_mass = sum(new_top_vals)
                bumped_out = original_topk_set - new_topk_set
                total_bumped_out += len(bumped_out)
                total_mass_increase += (new_total_mass - original_total_mass)

                top_acts[b, t] = torch.tensor(new_top_vals, device=device, dtype=top_acts.dtype)
                top_indices[b, t] = torch.tensor(new_top_idxs, device=device, dtype=top_indices.dtype)

        print(f"[DEBUG SAE] Neurons bumped out of Top-K: {total_bumped_out} | "
              f"Total Top-K mass increase: {total_mass_increase:.6f}")

        decoded = sae.decode(top_acts, top_indices)
        return decoded.to(dtype)

    return hook


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = get_args()
    logger = config.get_logger()

    # ---- model ----
    loader = m_loader.HFModelLoader(
        args.model_path, "llm", args.device, logger
    )
    model = loader.model
    tokenizer = loader.tokenizer
    model.eval()

    # ---- determine layers ----
    if args.layer is None:
        # Default to all layers
        num_layers = model.config.num_hidden_layers
        layer_indices = list(range(num_layers))
        layer_name_for_dir = "all"
        print(f"[INFO] No layers specified, targeting all {num_layers} layers")
    else:
        layer_indices = []
        for l in args.layer:
            # handle cases like "8", "layers.8", "model.layers.8.mlp"
            parts = l.split(".")
            for p in parts:
                if p.isdigit():
                    layer_indices.append(int(p))
                    break
        layer_name_for_dir = "_".join(map(str, sorted(layer_indices)))
        if len(layer_indices) > 5:
            layer_name_for_dir = f"multi_{len(layer_indices)}"
        print(f"[INFO] Targeting layers: {layer_indices}")

    rep_mode = "raw" if args.raw_model else "sae"

    # ---- layer configs ----
    layer_configs = []
    all_neuron_ids = {}
    
    # Pre-load SAEs if needed
    saes = {}
    if not args.raw_model:
        if args.sae_model is None:
            raise ValueError("sae-model is required when not in raw-model mode")
        # Construct layer strings for SAELoader
        layer_strings = [f"layers.{i}" for i in layer_indices]
        sae_manager = m_loader.SAELoader(
            args.sae_model, 
            layer_strings, 
            args.device, 
            logger
        )
        saes = sae_manager.sae_model # dict {layer_string: sae_model}

    for l_idx in layer_indices:
        # Resolve hook point
        layer_path = f"layers.{l_idx}"
        if args.raw_model:
            if "llama" in args.model_name.lower() or "gemma" in args.model_name.lower():
                layer_path = f"{layer_path}.mlp.act_fn"
        
        full_layer_path = f"model.{layer_path}"
        parts = full_layer_path.split(".")
        target_module = model
        for part in parts:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)
        
        # Load neurons
        if args.num_random_neurons is not None:
            neuron_ids = sample_random_neuron_ids(
                args.num_random_neurons, model, args.model_name, args.raw_model, saes.get(f"layers.{l_idx}"), args.seed + l_idx
            )
            print(f"[INFO] Sampled {len(neuron_ids)} random neurons for layer {l_idx}")
        else:
            neuron_ids = load_neurons_from_list(args, l_idx)
        
        all_neuron_ids[l_idx] = neuron_ids
        
        layer_configs.append({
            'idx': l_idx,
            'path': full_layer_path,
            'module': target_module,
            'neuron_ids': neuron_ids,
            'sae': saes.get(f"layers.{l_idx}")
        })

    # ---- output dir ----
    mode_name = f"random_{args.num_random_neurons}" if args.num_random_neurons is not None else args.category
    out_dir = os.path.join(
        paths.PROJECT_ROOT,
        args.output_root,
        args.model_name,
        rep_mode,
        f"{args.experiment_type}_{mode_name}",
        f"ablation_{args.ablation}_degree{args.max_degree if args.max_degree else 'all'}",
        f"layers_{layer_name_for_dir}",
        args.lang,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save neuron IDs
    with open(os.path.join(out_dir, "neuron_ids.json"), "w") as f:
        json.dump({
            "experiment": args.experiment_type,
            "category": args.category,
            "max_degree": args.max_degree,
            "layers": layer_indices,
            "neurons_per_layer": all_neuron_ids,
        }, f, indent=2)

    # ---- data ----
    dm = MultilingualDatasetManager(model_name=args.model_path)
    dl_target = dm.create_dataloader(
        "flores_plus", args.lang, args.split, batch_size=1, shuffle=False
    )
    print(f"Loaded {len(dl_target)} examples for {args.lang}")

    # ---- mean vectors ----
    mu_others = {} # {l_idx: vec}
    mu_currs = {}

    if args.ablation == "mean":
        if args.mean_source == "other":
            src_lang_other = args.mean_lang
            src_lang_curr = args.lang
            assert src_lang_other is not None
            
            print(f"Computing mean vectors for {src_lang_other} (other) across layers")
            dl_other = dm.create_dataloader("flores_plus", src_lang_other, args.split, batch_size=1, shuffle=False)
            mu_others = compute_mean_vectors(model, dl_other, layer_configs, args.device, args.raw_model)
            mu_others = {k: v.to(args.device) for k, v in mu_others.items()}

            print(f"Computing mean vectors for {src_lang_curr} (curr) across layers")
            dl_curr = dm.create_dataloader("flores_plus", src_lang_curr, args.split, batch_size=1, shuffle=False)
            mu_currs = compute_mean_vectors(model, dl_curr, layer_configs, args.device, args.raw_model)
            mu_currs = {k: v.to(args.device) for k, v in mu_currs.items()}
        else:
            print(f"Computing mean vectors for {args.lang} across layers")
            dl_mean = dm.create_dataloader("flores_plus", args.lang, args.split, batch_size=1, shuffle=False)
            mu_others = compute_mean_vectors(model, dl_mean, layer_configs, args.device, args.raw_model)
            mu_others = {k: v.to(args.device) for k, v in mu_others.items()}

    # ---- register hooks ----
    active_dict = {"active": False}
    handles = []
    for cfg in layer_configs:
        l_idx = cfg['idx']
        hook_fn = make_patch_hook(
            cfg['neuron_ids'], 
            mu_others.get(l_idx), 
            mu_currs.get(l_idx), 
            args.ablation, 
            args.raw_model, 
            cfg['sae'], 
            active_dict, 
            alpha=args.alpha, 
            mean_source=args.mean_source
        )
        h = cfg['module'].register_forward_hook(hook_fn)
        handles.append(h)

    # ---- evaluation ----
    rows = []
    generations = []
    for i, batch in enumerate(tqdm(dl_target)):
        if i >= args.max_examples:
            break

        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        with torch.no_grad():
            active_dict["active"] = False
            clean = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ppl_clean = compute_ppl(model, input_ids, attention_mask)
            
            clean_gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.gen_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            clean_text = tokenizer.decode(clean_gen[0][input_ids.shape[-1]:], skip_special_tokens=True)

            active_dict["active"] = True
            patched = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ppl_patch = compute_ppl(model, input_ids, attention_mask)
            
            patched_gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.gen_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            patched_text = tokenizer.decode(patched_gen[0][input_ids.shape[-1]:], skip_special_tokens=True)

        kl = kl_divergence(clean.logits[:, -1], patched.logits[:, -1]).item()
        rows.append({"idx": i, "kl": kl, "ppl_clean": ppl_clean, "ppl_patch": ppl_patch, "ppl_ratio": ppl_patch / max(ppl_clean, 1e-8)})
        generations.append({"idx": i, "prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True), "clean_text": clean_text, "patched_text": patched_text})

    # Cleanup hooks
    for h in handles:
        h.remove()

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(generations, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()

