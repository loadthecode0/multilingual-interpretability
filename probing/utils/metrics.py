import torch
import math
import bisect
from collections import Counter

def stack_activations_count(lang_to_stats, sorted_lang):
    """Stack activations count for each language."""
    print(f"\n[DEBUG] stack_activations_count called")
    print(f"[DEBUG] sorted_lang: {sorted_lang}")
    print(f"[DEBUG] lang_to_stats keys: {list(lang_to_stats.keys())}")
    
    num_examples = []
    num_tokens = []
    over_zero_token = []
    over_zero_example = []
    global_max_active_over_zero = None
    global_min_active_over_zero = None
    global_over_zero_total = 0
    global_over_zero_token = 0

    # Get feature dimension from first available tensor
    H = None
    for lang in sorted_lang:
        print(f"[DEBUG] Checking {lang} for feature dimension")
        if lang in lang_to_stats:
            print(f"[DEBUG]   {lang} has {len(lang_to_stats[lang])} layers")
            for i, layer in enumerate(lang_to_stats[lang]):
                print(f"[DEBUG]   Layer {i} over_zero_token is None: {layer['over_zero_token'] is None}")
                if layer["over_zero_token"] is not None:
                    H = layer["over_zero_token"].shape[0]
                    print(f"[DEBUG]   Found H = {H} from {lang} layer {i}")
                    break
        if H is not None:
            break
    
    if H is None:
        H = 16384  # Default SAE feature dimension
        print(f"[DEBUG] No feature dimension found, using default H = {H}")
    else:
        print(f"[DEBUG] Using feature dimension H = {H}")

    for lang in sorted_lang:
        print(f"\n[DEBUG] Processing language: {lang}")
        
        if lang not in lang_to_stats:
            print(f"[DEBUG] WARNING: {lang} not found in lang_to_stats!")
            continue
            
        # Language totals
        lang_num_examples = sum(layer["num_examples"] for layer in lang_to_stats[lang])
        lang_num_tokens = sum(layer["num_tokens"] for layer in lang_to_stats[lang])
        
        print(f"[DEBUG] {lang} totals: {lang_num_examples} examples, {lang_num_tokens} tokens")
        
        num_examples.append(lang_num_examples)
        num_tokens.append(lang_num_tokens)
        
        # Stack layers for this language
        lang_over_zero_token = []
        lang_over_zero_example = []
        lang_max_active = []
        lang_min_active = []
        
        for i, layer in enumerate(lang_to_stats[lang]):
            print(f"[DEBUG] Processing {lang} layer {i}")
            
            # Handle potential None values
            if layer["over_zero_token"] is not None:
                layer_token_count = layer["over_zero_token"].cpu()
                print(f"[DEBUG]   Layer {i} token_count shape: {layer_token_count.shape}, sum: {layer_token_count.sum().item()}")
                lang_over_zero_token.append(layer_token_count)
                # FIXED: Add to global_over_zero_token (matching Implementation 2)
                if isinstance(global_over_zero_token, int):
                    global_over_zero_token = layer_token_count.clone()
                else:
                    global_over_zero_token += layer_token_count
            else:
                print(f"[DEBUG]   Layer {i} token_count is None, using zeros")
                lang_over_zero_token.append(torch.zeros(H, dtype=torch.long))
                
            if layer["over_zero_example"] is not None:
                example_count = layer["over_zero_example"].cpu()
                print(f"[DEBUG]   Layer {i} example_count shape: {example_count.shape}, sum: {example_count.sum().item()}")
                lang_over_zero_example.append(example_count)
            else:
                print(f"[DEBUG]   Layer {i} example_count is None, using zeros")
                lang_over_zero_example.append(torch.zeros(H, dtype=torch.long))
                
            if layer["max_active_over_zero"] is not None:
                max_active = layer["max_active_over_zero"].cpu()
                print(f"[DEBUG]   Layer {i} max_active range: {max_active.min().item():.6f} to {max_active.max().item():.6f}")
                lang_max_active.append(max_active)
            else:
                print(f"[DEBUG]   Layer {i} max_active is None, using zeros")
                lang_max_active.append(torch.zeros(H, dtype=torch.float))
                
            if layer["min_active_over_zero"] is not None:
                min_active = layer["min_active_over_zero"].cpu()
                finite_count = (min_active != float('inf')).sum().item()
                print(f"[DEBUG]   Layer {i} min_active finite values: {finite_count}/{H}")
                if finite_count > 0:
                    finite_vals = min_active[min_active != float('inf')]
                    print(f"[DEBUG]   Layer {i} min_active finite range: {finite_vals.min().item():.6f} to {finite_vals.max().item():.6f}")
                lang_min_active.append(min_active)
            else:
                print(f"[DEBUG]   Layer {i} min_active is None, using inf")
                lang_min_active.append(torch.full((H,), float('inf'), dtype=torch.float))
            
            # Accumulate global totals
            if layer["over_zero_total"] is not None:
                total_count = layer["over_zero_total"].cpu()
                print(f"[DEBUG]   Layer {i} over_zero_total sum: {total_count.sum().item()}")
                if isinstance(global_over_zero_total, int):
                    global_over_zero_total = total_count.clone()
                else:
                    global_over_zero_total += total_count
        
        if not lang_over_zero_token:
            print(f"[DEBUG] WARNING: No layers processed for {lang}")
            continue
            
        # Stack tensors
        lang_over_zero_token = torch.stack(lang_over_zero_token)
        lang_over_zero_example = torch.stack(lang_over_zero_example)
        lang_max_active = torch.stack(lang_max_active)
        lang_min_active = torch.stack(lang_min_active)
        
        print(f"[DEBUG] {lang} stacked shapes:")
        print(f"[DEBUG]   over_zero_token: {lang_over_zero_token.shape}")
        print(f"[DEBUG]   over_zero_example: {lang_over_zero_example.shape}")
        print(f"[DEBUG]   max_active: {lang_max_active.shape}")
        print(f"[DEBUG]   min_active: {lang_min_active.shape}")
        
        over_zero_token.append(lang_over_zero_token)
        over_zero_example.append(lang_over_zero_example)
        
        # Update global max/min
        if global_max_active_over_zero is None:
            global_max_active_over_zero = lang_max_active
            print(f"[DEBUG] Initialized global_max_active_over_zero")
        else:
            global_max_active_over_zero = torch.maximum(global_max_active_over_zero, lang_max_active)
            print(f"[DEBUG] Updated global_max_active_over_zero")

        if global_min_active_over_zero is None:
            global_min_active_over_zero = lang_min_active
            print(f"[DEBUG] Initialized global_min_active_over_zero")
        else:
            global_min_active_over_zero = torch.minimum(global_min_active_over_zero, lang_min_active)
            print(f"[DEBUG] Updated global_min_active_over_zero")

    if not over_zero_token:
        print(f"[DEBUG] ERROR: No data collected for any language!")
        return None
    
    # Final stacking across languages
    num_examples = torch.tensor(num_examples, dtype=torch.long)
    num_tokens = torch.tensor(num_tokens, dtype=torch.long)
    over_zero_token = torch.stack(over_zero_token, dim=-1)  # (layers, hidden_dim, langs)
    over_zero_example = torch.stack(over_zero_example, dim=-1)

    print(f"[DEBUG] Final tensor shapes:")
    print(f"[DEBUG]   num_examples: {num_examples.shape}")
    print(f"[DEBUG]   num_tokens: {num_tokens.shape}")
    print(f"[DEBUG]   over_zero_token: {over_zero_token.shape}")
    print(f"[DEBUG]   over_zero_example: {over_zero_example.shape}")

    # Calculate global average
    if isinstance(global_over_zero_token, int):
        print(f"[DEBUG] WARNING: global_over_zero_token is still int: {global_over_zero_token}")
        global_avg_active_over_zero = torch.zeros_like(global_max_active_over_zero)
    else:
        print(f"[DEBUG] global_over_zero_total sum: {global_over_zero_total.sum().item()}")
        print(f"[DEBUG] global_over_zero_token sum: {global_over_zero_token.sum().item()}")
        global_avg_active_over_zero = global_over_zero_total / (global_over_zero_token + 1e-10)
        print(f"[DEBUG] global_avg_active_over_zero shape: {global_avg_active_over_zero.shape}")
        print(f"[DEBUG] global_avg_active_over_zero range: {global_avg_active_over_zero.min().item():.6f} to {global_avg_active_over_zero.max().item():.6f}")

    return (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    )

def stack_magnitude_stats(lang_to_stats, sorted_lang):
    """Stack magnitude statistics for each language (new function for magnitude ranking)."""
    print(f"\n[DEBUG] stack_magnitude_stats called")
    print(f"[DEBUG] sorted_lang: {sorted_lang}")
    print(f"[DEBUG] lang_to_stats keys: {list(lang_to_stats.keys())}")
    
    num_examples = []
    num_tokens = []
    activation_sums = []
    activation_squared_sums = []
    over_zero_token = []
    over_zero_example = []

    # Get feature dimension from first available tensor
    H = None
    for lang in sorted_lang:
        print(f"[DEBUG] Checking {lang} for feature dimension")
        if lang in lang_to_stats:
            print(f"[DEBUG]   {lang} has {len(lang_to_stats[lang])} layers")
            for i, layer in enumerate(lang_to_stats[lang]):
                if layer["activation_sum"] is not None:
                    H = layer["activation_sum"].shape[0]
                    print(f"[DEBUG]   Found H = {H} from {lang} layer {i}")
                    break
        if H is not None:
            break
    
    if H is None:
        H = 16384  # Default SAE feature dimension
        print(f"[DEBUG] No feature dimension found, using default H = {H}")
    else:
        print(f"[DEBUG] Using feature dimension H = {H}")

    for lang in sorted_lang:
        print(f"\n[DEBUG] Processing language: {lang}")
        
        if lang not in lang_to_stats:
            print(f"[DEBUG] WARNING: {lang} not found in lang_to_stats!")
            continue
            
        # Language totals
        lang_num_examples = sum(layer["num_examples"] for layer in lang_to_stats[lang])
        lang_num_tokens = sum(layer["num_tokens"] for layer in lang_to_stats[lang])
        
        print(f"[DEBUG] {lang} totals: {lang_num_examples} examples, {lang_num_tokens} tokens")
        
        num_examples.append(lang_num_examples)
        num_tokens.append(lang_num_tokens)
        
        # Stack layers for this language
        lang_activation_sums = []
        lang_activation_squared_sums = []
        lang_over_zero_token = []
        lang_over_zero_example = []
        
        for i, layer in enumerate(lang_to_stats[lang]):
            print(f"[DEBUG] Processing {lang} layer {i}")
            
            # Handle activation sums
            if layer["activation_sum"] is not None:
                activation_sum = layer["activation_sum"].cpu()
                print(f"[DEBUG]   Layer {i} activation_sum shape: {activation_sum.shape}, sum: {activation_sum.sum().item():.6f}")
                lang_activation_sums.append(activation_sum)
            else:
                print(f"[DEBUG]   Layer {i} activation_sum is None, using zeros")
                lang_activation_sums.append(torch.zeros(H, dtype=torch.float))
                
            # Handle activation squared sums
            if layer["activation_squared_sum"] is not None:
                activation_squared_sum = layer["activation_squared_sum"].cpu()
                print(f"[DEBUG]   Layer {i} activation_squared_sum shape: {activation_squared_sum.shape}, sum: {activation_squared_sum.sum().item():.6f}")
                lang_activation_squared_sums.append(activation_squared_sum)
            else:
                print(f"[DEBUG]   Layer {i} activation_squared_sum is None, using zeros")
                lang_activation_squared_sums.append(torch.zeros(H, dtype=torch.float))
                
            # Handle token counts (for filtering)
            if layer["over_zero_token"] is not None:
                token_count = layer["over_zero_token"].cpu()
                print(f"[DEBUG]   Layer {i} token_count shape: {token_count.shape}, sum: {token_count.sum().item()}")
                lang_over_zero_token.append(token_count)
            else:
                print(f"[DEBUG]   Layer {i} token_count is None, using zeros")
                lang_over_zero_token.append(torch.zeros(H, dtype=torch.long))
                
            # Handle example counts (for filtering)
            if layer["over_zero_example"] is not None:
                example_count = layer["over_zero_example"].cpu()
                print(f"[DEBUG]   Layer {i} example_count shape: {example_count.shape}, sum: {example_count.sum().item()}")
                lang_over_zero_example.append(example_count)
            else:
                print(f"[DEBUG]   Layer {i} example_count is None, using zeros")
                lang_over_zero_example.append(torch.zeros(H, dtype=torch.long))
        
        if not lang_activation_sums:
            print(f"[DEBUG] WARNING: No layers processed for {lang}")
            continue
            
        # Stack tensors
        lang_activation_sums = torch.stack(lang_activation_sums)
        lang_activation_squared_sums = torch.stack(lang_activation_squared_sums)
        lang_over_zero_token = torch.stack(lang_over_zero_token)
        lang_over_zero_example = torch.stack(lang_over_zero_example)
        
        print(f"[DEBUG] {lang} stacked shapes:")
        print(f"[DEBUG]   activation_sums: {lang_activation_sums.shape}")
        print(f"[DEBUG]   activation_squared_sums: {lang_activation_squared_sums.shape}")
        print(f"[DEBUG]   over_zero_token: {lang_over_zero_token.shape}")
        print(f"[DEBUG]   over_zero_example: {lang_over_zero_example.shape}")
        
        activation_sums.append(lang_activation_sums)
        activation_squared_sums.append(lang_activation_squared_sums)
        over_zero_token.append(lang_over_zero_token)
        over_zero_example.append(lang_over_zero_example)

    if not activation_sums:
        print(f"[DEBUG] ERROR: No data collected for any language!")
        return None
    
    # Final stacking across languages
    num_examples = torch.tensor(num_examples, dtype=torch.long)
    num_tokens = torch.tensor(num_tokens, dtype=torch.long)
    activation_sums = torch.stack(activation_sums, dim=-1)  # (layers, hidden_dim, langs)
    activation_squared_sums = torch.stack(activation_squared_sums, dim=-1)
    over_zero_token = torch.stack(over_zero_token, dim=-1)
    over_zero_example = torch.stack(over_zero_example, dim=-1)

    print(f"[DEBUG] Final tensor shapes:")
    print(f"[DEBUG]   num_examples: {num_examples.shape}")
    print(f"[DEBUG]   num_tokens: {num_tokens.shape}")
    print(f"[DEBUG]   activation_sums: {activation_sums.shape}")
    print(f"[DEBUG]   activation_squared_sums: {activation_squared_sums.shape}")
    print(f"[DEBUG]   over_zero_token: {over_zero_token.shape}")
    print(f"[DEBUG]   over_zero_example: {over_zero_example.shape}")

    return (
        num_examples,
        num_tokens,
        activation_sums,
        activation_squared_sums,
        over_zero_token,
        over_zero_example,
    )

def magnitude_ranking(
    num_examples,
    num_tokens,
    activation_sums,
    activation_squared_sums,
    over_zero_token,
    over_zero_example,
    sorted_lang,
    top=100,
    top_per_layer=False,
    apply_filtering=False
):
    """Magnitude-based ranking implementation following the original algorithm closely.
    
    Original algorithm:
    1. Calculate average activations per language
    2. For each language: avg_diff = lang_avg - mean(other_langs_avg)
    3. Sort features by activation difference (descending)
    4. Return top-k indices
    
    Args:
        apply_filtering: If True, applies SAE-LAPE style filtering. If False, uses all features.
    """
    print(f"\n[DEBUG] magnitude_ranking called")
    print(f"[DEBUG] Parameters:")
    print(f"[DEBUG]   top: {top}")
    print(f"[DEBUG]   top_per_layer: {top_per_layer}")
    print(f"[DEBUG]   apply_filtering: {apply_filtering}")
    print(f"[DEBUG]   sorted_lang: {sorted_lang}")
    
    num_layers, hidden_dim, num_langs = activation_sums.size()
    print(f"[DEBUG] Tensor dimensions: layers={num_layers}, hidden_dim={hidden_dim}, langs={num_langs}")
    
    print(f"[DEBUG] Input data summary:")
    print(f"[DEBUG]   num_examples: {num_examples}")
    print(f"[DEBUG]   num_tokens: {num_tokens}")
    print(f"[DEBUG]   activation_sums sum per lang: {activation_sums.sum(dim=(0,1))}")

    # Calculate average activations per language - equivalent to original avg_act
    print(f"\n[DEBUG] Calculating average activations...")
    avg_activations = activation_sums.float() / num_tokens.float().unsqueeze(0).unsqueeze(0)
    print(f"[DEBUG] avg_activations shape: {avg_activations.shape}")
    print(f"[DEBUG] avg_activations min: {avg_activations.min().item():.8f}")
    print(f"[DEBUG] avg_activations max: {avg_activations.max().item():.8f}")
    print(f"[DEBUG] avg_activations mean: {avg_activations.mean().item():.8f}")
    
    # Optional filtering (can be disabled to match original exactly)
    valid_features = None
    if apply_filtering:
        print(f"\n[DEBUG] Applying feature filtering...")
        
        # Example rate filtering
        example_rate = 0.98
        num_examples_thresh = (num_examples.float() * example_rate).long()
        print(f"[DEBUG] num_examples_thresh: {num_examples_thresh}")
        
        over_zero_example_filter = (over_zero_example >= num_examples_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        print(f"[DEBUG] Features passing example filter: {over_zero_example_filter.sum().item()}/{over_zero_example_filter.numel()}")

        # Token rate filtering
        hfl_rate = 0.1
        num_tokens_thresh = (num_tokens.float() * hfl_rate).long()
        print(f"[DEBUG] num_tokens_thresh: {num_tokens_thresh}")
        
        over_zero_token_filter = (over_zero_token > num_tokens_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        print(f"[DEBUG] Features passing token filter: {over_zero_token_filter.sum().item()}/{over_zero_token_filter.numel()}")

        # Apply filters
        valid_features = over_zero_example_filter & over_zero_token_filter
        print(f"[DEBUG] valid_features (passing both filters): {valid_features.sum().item()}/{valid_features.numel()}")
    else:
        print(f"\n[DEBUG] Skipping filtering - using all features (original algorithm behavior)")
    
    # Calculate activation differences for each language (matches original algorithm)
    final_indices = []
    features_info = {}
    
    for lang_idx, lang in enumerate(sorted_lang):
        print(f"\n[DEBUG] Processing {lang} (index {lang_idx})...")
        
        # Original algorithm: avg_act_difference_per_lan = avg_act_per_lan[i] - torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]], dim=0).mean(dim=0)
        other_langs_mask = torch.ones(num_langs, dtype=torch.bool)
        other_langs_mask[lang_idx] = False
        
        this_lang_activations = avg_activations[:, :, lang_idx]  # (layers, hidden_dim)
        other_langs_activations = avg_activations[:, :, other_langs_mask].mean(dim=-1)  # (layers, hidden_dim)
        
        activation_differences = this_lang_activations - other_langs_activations
        print(f"[DEBUG] {lang} activation_differences shape: {activation_differences.shape}")
        print(f"[DEBUG] {lang} activation_differences range: {activation_differences.min().item():.6f} to {activation_differences.max().item():.6f}")
        
        # Apply filtering only if requested
        if apply_filtering and valid_features is not None:
            activation_differences[~valid_features] = -float('inf')
            print(f"[DEBUG] {lang} valid activation_differences after filtering: {(activation_differences != -float('inf')).sum().item()}")
        
        # Flatten and sort (matches original: torch.sort(avg_act_difference_per_lan, descending=True))
        flattened_diffs = activation_differences.flatten()
        
        if apply_filtering:
            valid_mask = flattened_diffs != -float('inf')
            if valid_mask.sum() == 0:
                print(f"[DEBUG] No valid features for {lang}")
                final_indices.append([torch.tensor([], dtype=torch.long) for _ in range(num_layers)])
                features_info[lang] = {"indices": [], "avg_activations": torch.tensor([])}
                continue
            valid_diffs = flattened_diffs[valid_mask]
            valid_indices = torch.where(valid_mask)[0]
        else:
            # No filtering - use all features
            valid_diffs = flattened_diffs
            valid_indices = torch.arange(len(flattened_diffs))
        
        # Sort by activation difference (descending) - matches original
        sorted_diffs, sort_order = valid_diffs.sort(descending=True)
        sorted_indices = valid_indices[sort_order]
        
        print(f"[DEBUG] {lang} sorted_diffs range: {sorted_diffs.min().item():.6f} to {sorted_diffs.max().item():.6f}")
        
        # Convert to layer/feature coordinates
        layer_indices = sorted_indices // hidden_dim
        feature_indices = sorted_indices % hidden_dim
        
        print(f"[DEBUG] {lang} layer_indices range: {layer_indices.min().item()} to {layer_indices.max().item()}")
        print(f"[DEBUG] {lang} feature_indices range: {feature_indices.min().item()} to {feature_indices.max().item()}")
        
        # Apply top-k selection (this is an addition to the original but useful)
        lang_coords = list(zip(layer_indices.tolist(), feature_indices.tolist()))
        
        if top and len(lang_coords) > 0:
            print(f"[DEBUG] Applying top-{top} filter...")
            if top_per_layer:
                layer_counts = [0] * num_layers
                filtered_coords = []
                for layer_idx, feat_idx in lang_coords:
                    if layer_counts[layer_idx] < top:
                        filtered_coords.append((layer_idx, feat_idx))
                        layer_counts[layer_idx] += 1
                lang_coords = filtered_coords
                print(f"[DEBUG] After top-per-layer filter: {len(lang_coords)} features")
            else:
                lang_coords = lang_coords[:top]
                print(f"[DEBUG] After top filter: {len(lang_coords)} features")
        
        # Organize by layer
        layer_features = [[] for _ in range(num_layers)]
        for layer_idx, feat_idx in lang_coords:
            layer_features[layer_idx].append(feat_idx)
        
        # Convert to tensors
        for layer_idx in range(num_layers):
            layer_features[layer_idx] = torch.tensor(layer_features[layer_idx], dtype=torch.long)
            if len(layer_features[layer_idx]) > 0:
                print(f"[DEBUG] {lang} layer {layer_idx}: {len(layer_features[layer_idx])} features")
        
        final_indices.append(layer_features)
        
        # Store feature info
        if lang_coords:
            # Get the activation differences for selected features
            selected_diffs = []
            for layer_idx, feat_idx in lang_coords:
                if apply_filtering and valid_features is not None:
                    diff_val = activation_differences[layer_idx, feat_idx].item()
                else:
                    diff_val = activation_differences[layer_idx, feat_idx].item()
                selected_diffs.append(diff_val)
            
            features_info[lang] = {
                "indices": lang_coords,
                "avg_activations": torch.tensor(selected_diffs, dtype=torch.float)
            }
            print(f"[DEBUG] {lang} feature info: {len(lang_coords)} features stored")
        else:
            features_info[lang] = {"indices": [], "avg_activations": torch.tensor([])}

    print(f"\n[DEBUG] magnitude_ranking completed")
    total_features = sum(len(info["indices"]) for info in features_info.values())
    print(f"[DEBUG] Total features across all languages: {total_features}")
    
    return final_indices, features_info

def sae_lape(
    num_examples,
    num_tokens,
    over_zero_token,
    over_zero_example,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    sorted_lang,
    topk_threshold_ratio=0.8,
    example_rate=0.98,
    top=None,
    top_per_layer=False,
    entropy_threshold=None,
    lang_specific=True,
    lang_shared=False,
    shared_count=2,
    top_by_frequency=False
):
    """Original SAE-LAPE implementation."""
    print(f"\n[DEBUG] sae_lape called")
    print(f"[DEBUG] Parameters:")
    print(f"[DEBUG]   topk_threshold_ratio: {topk_threshold_ratio}")
    print(f"[DEBUG]   example_rate: {example_rate}")
    print(f"[DEBUG]   top: {top}")
    print(f"[DEBUG]   lang_specific: {lang_specific}")
    print(f"[DEBUG]   sorted_lang: {sorted_lang}")
    
    num_layers, hidden_dim, num_langs = over_zero_token.size()
    print(f"[DEBUG] Tensor dimensions: layers={num_layers}, hidden_dim={hidden_dim}, langs={num_langs}")
    
    print(f"[DEBUG] Input data summary:")
    print(f"[DEBUG]   num_examples: {num_examples}")
    print(f"[DEBUG]   num_tokens: {num_tokens}")
    print(f"[DEBUG]   over_zero_token sum per lang: {over_zero_token.sum(dim=(0,1))}")

    # Calculate activation probabilities
    print(f"\n[DEBUG] Calculating activation probabilities...")
    activation_probs = over_zero_token.float() / num_tokens.float()
    print(f"[DEBUG] activation_probs shape: {activation_probs.shape}")
    print(f"[DEBUG] activation_probs min: {activation_probs.min().item():.8f}")
    print(f"[DEBUG] activation_probs max: {activation_probs.max().item():.8f}")
    print(f"[DEBUG] activation_probs mean: {activation_probs.mean().item():.8f}")
    print(f"[DEBUG] Non-zero activation_probs: {(activation_probs > 0).sum().item()}/{activation_probs.numel()}")

    # L1 normalization
    print(f"\n[DEBUG] L1 normalization...")
    prob_sums = activation_probs.sum(dim=-1, keepdim=True)
    print(f"[DEBUG] prob_sums shape: {prob_sums.shape}")
    print(f"[DEBUG] prob_sums min: {prob_sums.min().item():.8f}")
    print(f"[DEBUG] prob_sums max: {prob_sums.max().item():.8f}")
    print(f"[DEBUG] Zero prob_sums: {(prob_sums == 0).sum().item()}")
    
    normed_activation_probs = activation_probs / (prob_sums + 1e-10)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    print(f"[DEBUG] normed_activation_probs shape: {normed_activation_probs.shape}")
    print(f"[DEBUG] normed_activation_probs min: {normed_activation_probs.min().item():.8f}")
    print(f"[DEBUG] normed_activation_probs max: {normed_activation_probs.max().item():.8f}")
    print(f"[DEBUG] normed_activation_probs sum along lang dim (should be ~1): {normed_activation_probs.sum(dim=-1).mean().item():.8f}")
    # valid = prob_sums.squeeze(-1) > 0
    # print("Mean over valid neurons only:",
    #     normed_activation_probs[valid].view(-1, 2).sum(dim=-1).mean().item())
    
    # Entropy calculation
    print(f"\n[DEBUG] Calculating entropy...")
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    print(f"[DEBUG] log_probs min: {log_probs.min().item():.6f}")
    print(f"[DEBUG] log_probs max: {log_probs.max().item():.6f}")
    print(f"[DEBUG] Non-zero log_probs: {(log_probs != 0).sum().item()}/{log_probs.numel()}")
    
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    print(f"[DEBUG] entropy shape: {entropy.shape}")
    print(f"[DEBUG] entropy min: {entropy.min().item():.6f}")
    print(f"[DEBUG] entropy max: {entropy.max().item():.6f}")
    print(f"[DEBUG] entropy mean: {entropy.mean().item():.6f}")
    print(f"[DEBUG] Non-zero entropy: {(entropy > 0).sum().item()}/{entropy.numel()}")

    # Feature filtering
    largest = False  # We want smallest entropy (most language-specific)
    print(f"\n[DEBUG] Feature filtering (largest={largest})...")

    # Example rate filtering
    num_examples_thresh = (num_examples.float() * example_rate).long()
    print(f"[DEBUG] num_examples_thresh: {num_examples_thresh}")
    
    over_zero_example_filter = (over_zero_example >= num_examples_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)
    print(f"[DEBUG] over_zero_example_filter shape: {over_zero_example_filter.shape}")
    print(f"[DEBUG] Features passing example filter: {over_zero_example_filter.sum().item()}/{over_zero_example_filter.numel()}")

    # Token rate filtering
    hfl_rate = 0.1
    num_tokens_thresh = (num_tokens.float() * hfl_rate).long()
    print(f"[DEBUG] num_tokens_thresh: {num_tokens_thresh}")
    
    over_zero_token_filter = (over_zero_token > num_tokens_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)
    print(f"[DEBUG] over_zero_token_filter shape: {over_zero_token_filter.shape}")
    print(f"[DEBUG] Features passing token filter: {over_zero_token_filter.sum().item()}/{over_zero_token_filter.numel()}")

    # Apply filters
    dismissed_neurons = over_zero_example_filter & over_zero_token_filter
    print(f"[DEBUG] dismissed_neurons (passing both filters): {dismissed_neurons.sum().item()}/{dismissed_neurons.numel()}")
    
    patched_val = torch.inf if not largest else -torch.inf
    print(f"[DEBUG] patched_val: {patched_val}")
    
    entropy_before_patch = entropy.clone()
    entropy[~dismissed_neurons] = patched_val
    
    print(f"[DEBUG] Entropy after patching:")
    print(f"[DEBUG]   Non-patched values: {(entropy != patched_val).sum().item()}")
    print(f"[DEBUG]   Valid entropy range: {entropy[entropy != patched_val].min().item():.6f} to {entropy[entropy != patched_val].max().item():.6f}")

    # Select features by entropy
    print(f"\n[DEBUG] Selecting features by entropy...")
    flattened_entropy = entropy.flatten()
    valid_mask = flattened_entropy != patched_val
    
    print(f"[DEBUG] flattened_entropy shape: {flattened_entropy.shape}")
    print(f"[DEBUG] valid_mask sum: {valid_mask.sum().item()}")
    
    if valid_mask.sum() == 0:
        print(f"[DEBUG] ERROR: No valid features found!")
        return [], {}

    valid_entropies = flattened_entropy[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    print(f"[DEBUG] valid_entropies shape: {valid_entropies.shape}")
    print(f"[DEBUG] valid_entropies range: {valid_entropies.min().item():.6f} to {valid_entropies.max().item():.6f}")
    
    # Sort by entropy
    sorted_entropies, sort_order = valid_entropies.sort()
    sorted_indices = valid_indices[sort_order]
    
    print(f"[DEBUG] sorted_entropies range: {sorted_entropies.min().item():.6f} to {sorted_entropies.max().item():.6f}")

    # Convert to layer/feature coordinates
    layer_indices = sorted_indices // hidden_dim
    feature_indices = sorted_indices % hidden_dim
    
    print(f"[DEBUG] layer_indices range: {layer_indices.min().item()} to {layer_indices.max().item()}")
    print(f"[DEBUG] feature_indices range: {feature_indices.min().item()} to {feature_indices.max().item()}")
    
    # Get probabilities for selected features
    selected_probs = activation_probs[layer_indices, feature_indices]
    selected_probs = selected_probs.transpose(0, 1)  # (langs, features)
    
    print(f"[DEBUG] selected_probs shape: {selected_probs.shape}")
    print(f"[DEBUG] selected_probs range: {selected_probs.min().item():.8f} to {selected_probs.max().item():.8f}")
    
    # Language assignment
    print(f"\n[DEBUG] Language assignment...")
    max_probs = selected_probs.max(dim=0, keepdim=True)[0]
    print(f"[DEBUG] max_probs shape: {max_probs.shape}")
    print(f"[DEBUG] max_probs range: {max_probs.min().item():.8f} to {max_probs.max().item():.8f}")
    
    threshold = max_probs * topk_threshold_ratio
    print(f"[DEBUG] threshold range: {threshold.min().item():.8f} to {threshold.max().item():.8f}")
    
    lang_mask = selected_probs >= threshold
    print(f"[DEBUG] lang_mask shape: {lang_mask.shape}")
    print(f"[DEBUG] lang_mask true values: {lang_mask.sum().item()}/{lang_mask.numel()}")
    
    # Handle sharing preferences
    feature_counts = lang_mask.sum(dim=0)
    print(f"[DEBUG] feature_counts shape: {feature_counts.shape}")
    print(f"[DEBUG] feature_counts range: {feature_counts.min().item()} to {feature_counts.max().item()}")
    print(f"[DEBUG] feature_counts distribution:")
    for i in range(num_langs + 1):
        count = (feature_counts == i).sum().item()
        if count > 0:
            print(f"[DEBUG]   {count} features assigned to {i} languages")


    # --- Collect shared features BEFORE modifying lang_mask ---
    merged_coords = torch.stack([layer_indices, feature_indices], dim=1)
    shared_features = []
    for i in range(len(sorted_entropies)):
        count = feature_counts[i].item()
        print(count)
        if count > 1:  # neuron shared across multiple languages
            layer, feat = merged_coords[i].tolist()
            langs = [sorted_lang[j] for j in range(num_langs) if lang_mask[j, i]]
            shared_features.append({
                # "layer": layer,
                "feature_idx": feat,
                "languages": langs,
                "num_languages": len(langs),
                "entropy": sorted_entropies[i].item()
            })

    # print(shared_features)

    
    if lang_specific:
        print(f"[DEBUG] Applying lang_specific filter...")
        # Remove shared features
        shared_mask = feature_counts > 1
        print(f"[DEBUG] Removing {shared_mask.sum().item()} shared features")
        lang_mask[:, shared_mask] = False
    elif lang_shared:
        print(f"[DEBUG] Applying lang_shared filter (shared_count={shared_count})...")
        # Keep only features shared across exactly shared_count languages
        wrong_count_mask = feature_counts != shared_count
        print(f"[DEBUG] Removing {wrong_count_mask.sum().item()} features not shared by exactly {shared_count} languages")
        lang_mask[:, wrong_count_mask] = False
    
    print(f"[DEBUG] After sharing filter:")
    remaining_features = lang_mask.sum().item()
    print(f"[DEBUG] Remaining assignments: {remaining_features}")
    
    # Extract results per language
    final_indices = []
    features_info = {}
    merged_coords = torch.stack([layer_indices, feature_indices], dim=1)
    
    print(f"\n[DEBUG] Extracting results per language...")
    
    for lang_idx, lang in enumerate(sorted_lang):
        print(f"\n[DEBUG] Processing {lang} (index {lang_idx})...")
        
        lang_feature_mask = lang_mask[lang_idx]
        print(f"[DEBUG] {lang} feature mask sum: {lang_feature_mask.sum().item()}")
        
        selected_coords = merged_coords[lang_feature_mask]
        print(f"[DEBUG] {lang} selected_coords shape: {selected_coords.shape}")
        
        if len(selected_coords) == 0:
            print(f"[DEBUG] No features for {lang}")
            final_indices.append([torch.tensor([], dtype=torch.long) for _ in range(num_layers)])
            features_info[lang] = {"indices": [], "selected_probs": torch.tensor([]), "entropies": torch.tensor([])}
            continue
            
        # Convert to coordinate tuples
        lang_coords = [tuple(coord.tolist()) for coord in selected_coords]
        print(f"[DEBUG] {lang} coordinates: {len(lang_coords)} features")
        
        # Apply top-k if specified
        if top and len(lang_coords) > 0:
            print(f"[DEBUG] Applying top-{top} filter...")
            if top_per_layer:
                layer_counts = [0] * num_layers
                filtered_coords = []
                for layer_idx, feat_idx in lang_coords:
                    if layer_counts[layer_idx] < top:
                        filtered_coords.append((layer_idx, feat_idx))
                        layer_counts[layer_idx] += 1
                lang_coords = filtered_coords
                print(f"[DEBUG] After top-per-layer filter: {len(lang_coords)} features")
            else:
                lang_coords = lang_coords[:top]
                print(f"[DEBUG] After top filter: {len(lang_coords)} features")
        
        # Organize by layer
        layer_features = [[] for _ in range(num_layers)]
        for layer_idx, feat_idx in lang_coords:
            layer_features[layer_idx].append(feat_idx)
        
        # Convert to tensors
        for layer_idx in range(num_layers):
            layer_features[layer_idx] = torch.tensor(layer_features[layer_idx], dtype=torch.long)
            if len(layer_features[layer_idx]) > 0:
                print(f"[DEBUG] {lang} layer {layer_idx}: {len(layer_features[layer_idx])} features")
        
        final_indices.append(layer_features)
        
        # Store feature info
        if lang_coords:
            coord_indices = []
            for coord in lang_coords:
                coord_tensor = torch.tensor(coord)
                matches = (merged_coords == coord_tensor).all(dim=1)
                coord_indices.extend(torch.where(matches)[0].tolist())
            
            features_info[lang] = {
                "indices": lang_coords,
                "selected_probs": selected_probs[lang_idx, coord_indices] if coord_indices else torch.tensor([]),
                "entropies": sorted_entropies[coord_indices] if coord_indices else torch.tensor([])
            }
            print(f"[DEBUG] {lang} feature info: {len(coord_indices)} features stored")
        else:
            features_info[lang] = {"indices": [], "selected_probs": torch.tensor([]), "entropies": torch.tensor([])}

    # # Collect shared features explicitly (fix: align with selected features)
    # shared_features = []

    # for i in range(len(sorted_entropies)):
    #     count = feature_counts[i].item()
    #     if count > 1:  # neuron active in >1 language
    #         layer, feat = merged_coords[i].tolist()
    #         langs = [sorted_lang[j] for j in range(num_langs) if lang_mask[j, i]]
    #         shared_features.append({
    #             "layer": layer,
    #             "feature_idx": feat,
    #             "languages": langs,
    #             "num_languages": len(langs),
    #             "entropy": sorted_entropies[i].item()
    #         })

    # print(f"[DEBUG] Found {len(shared_features)} shared features")

    return final_indices, features_info, shared_features

    print(f"\n[DEBUG] sae_lape completed")
    total_features = sum(len(info["indices"]) for info in features_info.values())
    print(f"[DEBUG] Total features across all languages: {total_features}")
    
    return final_indices, features_info