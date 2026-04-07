#!/usr/bin/env python3
"""
Unified causal intervention script.

Supports:
- SAE or raw hidden-state interventions
- Zero / mean ablation
- Mean from same or other language
- KL divergence + PPL metrics
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
    p.add_argument("--model_name", required=True)
    p.add_argument("--sae-model", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--raw-model", action="store_true")

    # neurons
    p.add_argument("--dominance-file", required=True)
    p.add_argument("--feature-set", required=True)
    p.add_argument("--top-k", type=int, default=50)

    # ablation
    p.add_argument("--ablation", choices=["zero", "mean", "null"], required=True)
    p.add_argument("--mean-source", choices=["same", "other"], default="same")
    p.add_argument("--mean-lang", default=None)

    # data
    p.add_argument("--lang", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--max-examples", type=int, default=20)

    # output
    p.add_argument("--output-root", default="causal_results")
    p.add_argument("--gen-tokens", type=int, default=20)

    p.add_argument(
        "--neuron-mode",
        choices=["dominant", "random"],
        default="dominant",
        help="Use dominant neurons or a matched random control set"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for random neuron selection"
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Tunable argument for cross-ablation: h + alpha(mu_other - mu_curr)"
    )

    return p.parse_args()


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_topk_neurons(csv_path, k):
    df = pd.read_csv(csv_path)
    df = df.sort_values("rank", ascending=True)
    topk_list = df["neuron_idx"].head(k).tolist()
    print(f"Loaded {len(topk_list)} neurons from {csv_path}, example neurons: {topk_list[:5]}")
    return topk_list

def sample_random_neurons(
    dominance_csv: str,
    model_name: str,
    raw_model: bool,
    k: int,
    seed: int,
):
    """
    Sample k neurons uniformly at random from the complement
    of the dominant neuron set.
    """
    df = pd.read_csv(dominance_csv)

    dominant = set(df["neuron_idx"].tolist())

    # all_neurons = df["neuron_idx"].unique().tolist()
    if "llama" in model_name.lower():
        if raw_model:
            num_neurons = 8192 # Intermediate size for Llama-3.2-1B
        else:
            num_neurons = 131072
    elif "gemma" in model_name.lower():
        if raw_model:
            num_neurons = 9216 # Intermediate size for Gemma-2-2b
        else:
            num_neurons = 65536
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    pool = [n for n in range(num_neurons) if n not in dominant]

    assert len(pool) >= k, "Not enough neurons to sample random control"

    rng = np.random.default_rng(seed)
    random_neurons = rng.choice(pool, size=k, replace=False).tolist()

    return random_neurons



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
def compute_mean_vector(
    model, sae, dataloader, layer_idx, device, raw_model, max_batches=100
):
    acc = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = out.hidden_states[layer_idx + 1]

        if raw_model:
            acts = hidden
        else:
            sae_out = sae.encode(hidden)
            acts = sae_out.pre_acts

        acc.append(acts.mean(dim=(0, 1)))

    return torch.stack(acc).mean(dim=0)


# -------------------------------------------------
# Hooks
# -------------------------------------------------
# def make_patch_hook(neuron_ids, mean_vec, ablation, raw_model, sae, active_dict):
#     def hook(module, inp, out):
#         if not active_dict["active"]:
#             return out

#         device = out.device
#         dtype = out.dtype
#         idx = torch.tensor(neuron_ids, device=device)

#         if raw_model:
#             h = out.clone()
#             if ablation == "zero":
#                 h[:, :, idx] = 0.0
#             else:
#                 if mean_vec is None:
#                     raise ValueError("mean_vec must be provided for mean ablation")
#                 h[:, :, idx] = mean_vec.to(device)[idx]
#             return h.to(dtype)

#         else:
#             # out is the model activation, already on device.
#             # sae should have been moved to device in main.
#             sae_out = sae.encode(out)
#             # lat = sae_out.pre_acts
#             pre = sae_out.pre_acts
#             if ablation == "zero":
#                 pre[:, :, idx] = 0.0
#             else:
#                 if mean_vec is None:
#                     raise ValueError("mean_vec must be provided for mean ablation")
#                 pre[:, :, idx] = mean_vec.to(device)[idx]
#             # Re-apply the SAE sparsifier
#             # top_values, top_indices = sae.topk(pre)
#             # lat = sae.decode(top_values, top_indices)
#             lat = sae.decode(sae_out.top_acts, sae_out.top_indices)
#             return lat.to(dtype)

#     return hook
# -------------------------------------------------
# Hook for causal intervention (TopK SAE compatible)
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
    """
    Causal hook supporting:
    - raw-model ablation
    - SparseCoder Top-K post-selection
    - Counterfactual Top-K membership override
    - Cross-language: h + alpha(mu_other - mu_curr)
    """

    neuron_ids = list(neuron_ids)

    def hook(module, inp, out):
        if not active_dict["active"]:
            return out
        
        if ablation == "null":
            print(f"Null ablation (no-op)")
            return out

        device = out.device
        dtype = out.dtype

        # ======================================================
        # RAW MODEL (direct hidden-state ablation)
        # ======================================================
        if raw_model:
            h = out.clone()

            idx = torch.tensor(neuron_ids, device=device)
            old_vals = out[:, :, idx].clone()

            if ablation == "zero":
                h[:, :, idx] = 0.0
            else:
                if mean_source == "other":
                    # h' = h + alpha(mu_other - mu_curr)
                    diff = (mu_other[idx] - mu_curr[idx]).to(device=device, dtype=h.dtype)
                    h[:, :, idx] = h[:, :, idx] + alpha * diff
                else:
                    h[:, :, idx] = mu_other[idx].to(
                        device=device,
                        dtype=h.dtype,
                    )
            
            new_vals = h[:, :, idx]
            diff_vals = (new_vals - old_vals).abs()
            num_changed = (diff_vals > 1e-5).sum().item()
            avg_increase = (new_vals - old_vals).mean().item()
            print(f"[DEBUG RAW] Neurons changed: {num_changed} | Avg value increase: {avg_increase:.6f}")

            return h.to(dtype)

        # ======================================================
        # SPARSECODER (Top-K membership intervention)
        # ======================================================
        enc = sae.encode(out)

        top_acts = enc.top_acts.clone()        # (B, T, K)
        top_indices = enc.top_indices.clone()  # (B, T, K)

        B, T, K = top_acts.shape

        total_bumped_out = 0
        total_mass_increase = 0.0

        for b in range(B):
            for t in range(T):
                acts_bt = top_acts[b, t]       # (K,)
                idxs_bt = top_indices[b, t]    # (K,)

                # Collect candidates for new Top-K
                # Start with current Top-K entries
                candidate_vals = acts_bt.tolist()
                candidate_idxs = idxs_bt.tolist()
                
                original_topk_set = set(candidate_idxs)
                original_total_mass = sum(candidate_vals)

                original_topk_vals = {idx: val for idx, val in zip(candidate_idxs, candidate_vals)}

                for nid in neuron_ids:
                    # 1. Determine current activation h_nid
                    h_nid = original_topk_vals.get(nid, 0.0)

                    # 2. Compute h_prime
                    if ablation == "zero":
                        h_prime = 0.0
                    elif ablation == "mean":
                        if mean_source == "other":
                            v_other = mu_other[nid].item()
                            v_curr = mu_curr[nid].item()
                            h_prime = h_nid + alpha * (v_other - v_curr)
                        else:
                            h_prime = mu_other[nid].item()
                        
                        # Clamp to non-negative for SAE activations
                        h_prime = max(0.0, h_prime)
                    else:
                        h_prime = h_nid

                    # 3. Update if already in Top-K, else add as candidate
                    if nid in original_topk_set:
                        for k_idx in range(K):
                            if candidate_idxs[k_idx] == nid:
                                candidate_vals[k_idx] = h_prime
                                break
                    else:
                        candidate_vals.append(h_prime)
                        candidate_idxs.append(nid)

                # 4. Select new Top-K from all candidates
                combined = sorted(zip(candidate_vals, candidate_idxs), key=lambda x: x[0], reverse=True)
                new_top_vals, new_top_idxs = zip(*combined[:K])
                
                new_topk_set = set(new_top_idxs)
                new_total_mass = sum(new_top_vals)

                # How many from the original Top-K are no longer there?
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

    layer_idx = int(args.layer.split(".")[1])
    rep_mode = "raw" if args.raw_model else "sae"

    # ---- output dir ----
    out_dir = os.path.join(
        paths.PROJECT_ROOT,
        args.output_root,
        args.model_name,
        rep_mode,
        f"ablation_{args.ablation}_{args.neuron_mode}",
        f"mean_from_{args.mean_source}_top{args.top_k}",
        args.feature_set,
        f"layer_{layer_idx}",
        args.lang,
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- model ----
    loader = m_loader.HFModelLoader(
        args.model_path, "llm", args.device, logger
    )
    model = loader.model
    tokenizer = loader.tokenizer
    model.eval()

    if not args.raw_model:
        sae = m_loader.SAELoader(
            args.sae_model, 
            [args.layer], 
            args.device, 
            logger
        ).sae_model[args.layer]
        sae.to(args.device)
    else:
        sae = None

    # ---- neurons ----
    # neuron_ids = load_topk_neurons(args.dominance_file, args.top_k)
    if args.neuron_mode == "dominant":
        neuron_ids = load_topk_neurons(args.dominance_file, args.top_k)
    else:
        neuron_ids = sample_random_neurons(
            args.dominance_file,
            args.model_name,
            args.raw_model,
            args.top_k,
            seed=args.seed,
        )
        print(
            f"[RANDOM MODE] Using random neurons (seed={args.seed}): "
            f"{neuron_ids[:5]} ..."
        )

    # Save neuron IDs for reproducibility
    with open(os.path.join(out_dir, "neuron_ids.json"), "w") as f:
        json.dump(
            {
                "mode": args.neuron_mode,
                "top_k": args.top_k,
                "seed": args.seed,
                "neurons": neuron_ids,
            },
            f,
            indent=2,
        )


    # ---- data ----
    dm = MultilingualDatasetManager(model_name=args.model_path)
    dl_target = dm.create_dataloader(
        "flores_plus",
        args.lang,
        args.split,
        batch_size=1,
        shuffle=False,
    )

    # ---- mean vector ----
    mu_other = None
    mu_curr = None

    if args.ablation == "mean":
        if args.mean_source == "other":
            src_lang_other = args.mean_lang
            src_lang_curr = args.lang
            assert src_lang_other is not None

            print(f"Computing mean vectors for cross-ablation: {src_lang_other} (other) and {src_lang_curr} (curr)")
            
            dl_other = dm.create_dataloader(
                "flores_plus", src_lang_other, args.split,
                batch_size=1, shuffle=False
            )
            mu_other = compute_mean_vector(
                model, sae, dl_other,
                layer_idx, args.device,
                args.raw_model,
            ).to(args.device)

            dl_curr = dm.create_dataloader(
                "flores_plus", src_lang_curr, args.split,
                batch_size=1, shuffle=False
            )
            mu_curr = compute_mean_vector(
                model, sae, dl_curr,
                layer_idx, args.device,
                args.raw_model,
            ).to(args.device)
        else:
            src_lang = args.lang
            print(f"Computing mean vector for same-language ablation: {src_lang}")
            dl_mean = dm.create_dataloader(
                "flores_plus", src_lang, args.split,
                batch_size=1, shuffle=False
            )
            mu_other = compute_mean_vector(
                model, sae, dl_mean,
                layer_idx, args.device,
                args.raw_model,
            ).to(args.device)

    # ---- hook ----
    active_dict = {"active": False}
    hook_fn = make_patch_hook(
        neuron_ids,
        mu_other,
        mu_curr,
        args.ablation,
        args.raw_model,
        sae,
        active_dict,
        alpha=args.alpha,
        mean_source=args.mean_source,
    )

    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)

    # ---- evaluation ----
    rows = []
    generations = []
    for i, batch in enumerate(tqdm(dl_target)):
        if i >= args.max_examples:
            break

        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        with torch.no_grad():
            # Clean run
            active_dict["active"] = False
            clean = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          return_dict=True)
            ppl_clean = compute_ppl(model, input_ids, attention_mask)
            
            clean_gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            clean_text = tokenizer.decode(clean_gen[0][input_ids.shape[-1]:], skip_special_tokens=True)

            # Patched run
            active_dict["active"] = True
            patched = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
            ppl_patch = compute_ppl(model, input_ids, attention_mask)
            
            patched_gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            patched_text = tokenizer.decode(patched_gen[0][input_ids.shape[-1]:], skip_special_tokens=True)

        kl = kl_divergence(
            clean.logits[:, -1],
            patched.logits[:, -1],
        ).item()

        rows.append({
            "idx": i,
            "kl": kl,
            "ppl_clean": ppl_clean,
            "ppl_patch": ppl_patch,
            "ppl_ratio": ppl_patch / max(ppl_clean, 1e-8),
        })
        
        generations.append({
            "idx": i,
            "prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True),
            "clean_text": clean_text,
            "patched_text": patched_text,
        })

    handle.remove()

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False
    )
    
    with open(os.path.join(out_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(generations, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved metrics and generations → {out_dir}")


if __name__ == "__main__":
    main()
