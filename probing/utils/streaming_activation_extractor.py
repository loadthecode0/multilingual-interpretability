import torch
import math
from tqdm import tqdm
from collections import defaultdict
from itertools import islice

class StreamingExtractor:
    def __init__(self, model, saes, device):
        self.model = model
        self.saes = saes
        self.device = device
        self.lang_to_stats = defaultdict(list)

    def update_stats(self, lang, sae_latents, layer_idx):
        """Update running counts - simplified version."""
        print(f"\n[DEBUG] update_stats called for {lang}, layer {layer_idx}")
        print(f"[DEBUG] sae_latents shape: {sae_latents.shape}")
        print(f"[DEBUG] sae_latents device: {sae_latents.device}")
        print(f"[DEBUG] sae_latents dtype: {sae_latents.dtype}")
        print(f"[DEBUG] sae_latents min: {sae_latents.min().item():.6f}, max: {sae_latents.max().item():.6f}")
        print(f"[DEBUG] sae_latents mean: {sae_latents.mean().item():.6f}")
        print(f"[DEBUG] Number of positive values: {(sae_latents > 0).sum().item()}")
        
        # Ensure enough layers allocated
        while len(self.lang_to_stats[lang]) <= layer_idx:
            self.lang_to_stats[lang].append({
                "num_examples": 0,
                "num_tokens": 0,
                "over_zero_token": None,
                "over_zero_example": None,
                "over_zero_total": None,
                "max_active_over_zero": None,
                "min_active_over_zero": None,
                # Additional fields for magnitude ranking
                "activation_sum": None,
                "activation_squared_sum": None,
            })

        stats = self.lang_to_stats[lang][layer_idx]
        B, T, H = sae_latents.shape
        print(f"[DEBUG] Batch size: {B}, Sequence length: {T}, Hidden dim: {H}")
        # In update_stats(), add:
        unique_values = torch.unique(sae_latents[sae_latents > 0])
        print(f"[DEBUG] Unique positive activation values (first 10): {unique_values[:10]}")
        # Initialize on first batch
        if stats["over_zero_token"] is None:
            print(f"[DEBUG] Initializing stats tensors for {lang} layer {layer_idx}")
            stats["over_zero_token"] = torch.zeros(H, dtype=torch.long, device=self.device)
            stats["over_zero_example"] = torch.zeros(H, dtype=torch.long, device=self.device)
            stats["over_zero_total"] = torch.zeros(H, dtype=torch.float, device=self.device)
            stats["max_active_over_zero"] = torch.zeros(H, dtype=torch.float, device=self.device)
            stats["min_active_over_zero"] = torch.full((H,), float('inf'), dtype=torch.float, device=self.device)
            # Initialize magnitude tracking
            stats["activation_sum"] = torch.zeros(H, dtype=torch.float, device=self.device)
            stats["activation_squared_sum"] = torch.zeros(H, dtype=torch.float, device=self.device)
        # Compute statistics for this batch
        over_zero_mask = sae_latents > 0
        print(f"[DEBUG] over_zero_mask shape: {over_zero_mask.shape}")
        print(f"[DEBUG] Total positive activations in batch: {over_zero_mask.sum().item()}")
        
        # Token-level counts
        token_counts = over_zero_mask.sum(dim=(0, 1))  # Sum over batch and sequence
        print(f"[DEBUG] token_counts shape: {token_counts.shape}")
        print(f"[DEBUG] token_counts min: {token_counts.min().item()}, max: {token_counts.max().item()}")
        print(f"[DEBUG] Features with >0 token activations: {(token_counts > 0).sum().item()}/{H}")
        
        # Example-level counts
        example_mask = over_zero_mask.sum(dim=1) > 0  # Any activation in sequence
        example_counts = example_mask.sum(dim=0)  # Sum over batch
        print(f"[DEBUG] example_counts shape: {example_counts.shape}")
        print(f"[DEBUG] example_counts min: {example_counts.min().item()}, max: {example_counts.max().item()}")
        print(f"[DEBUG] Features with >0 example activations: {(example_counts > 0).sum().item()}/{H}")
        
        # Magnitude statistics for activation averaging
        activation_sum = sae_latents.sum(dim=(0, 1))  # Sum over batch and sequence
        activation_squared_sum = (sae_latents ** 2).sum(dim=(0, 1))
        print(f"[DEBUG] activation_sum range: {activation_sum.min().item():.6f} to {activation_sum.max().item():.6f}")
        print(f"[DEBUG] activation_squared_sum range: {activation_squared_sum.min().item():.6f} to {activation_squared_sum.max().item():.6f}")
        
        # Max values
        batch_max = sae_latents.max()
        print(f"[DEBUG] batch_max: {batch_max.item():.6f}")
        
        if batch_max > 0:  # Only update if there are activations
            feature_max = sae_latents.view(-1, H).max(dim=0)[0]
            print(f"[DEBUG] feature_max shape: {feature_max.shape}")
            print(f"[DEBUG] feature_max range: {feature_max.min().item():.6f} to {feature_max.max().item():.6f}")
        else:
            feature_max = torch.zeros(H, dtype=torch.float, device=self.device)
            print(f"[DEBUG] No positive activations, using zero feature_max")
        
        # Min values (only for non-zero activations)
        feature_min = torch.full((H,), float('inf'), dtype=torch.float, device=self.device)
        # sae_flat = sae_latents.view(-1, H)
        
        # nonzero_feature_count = 0
        # for h in range(H):
        #     nonzero_values = sae_flat[:, h][sae_flat[:, h] > 0]
        #     if len(nonzero_values) > 0:
        #         feature_min[h] = nonzero_values.min().item()
        #         nonzero_feature_count += 1
        
        # print(f"[DEBUG] Features with nonzero values: {nonzero_feature_count}/{H}")
        # finite_mins = feature_min[feature_min != float('inf')]
        # if len(finite_mins) > 0:
        #     print(f"[DEBUG] feature_min range (finite): {finite_mins.min().item():.6f} to {finite_mins.max().item():.6f}")
        
        # Update accumulated statistics
        old_examples = stats["num_examples"]
        old_tokens = stats["num_tokens"]
        
        stats["num_examples"] += B
        stats["num_tokens"] += B * T
        stats["over_zero_token"] += token_counts
        stats["over_zero_example"] += example_counts
        # stats["over_zero_total"] += token_counts  # This seems redundant with over_zero_token
        positive_activations = sae_latents * over_zero_mask.float()
        stats["over_zero_total"] += positive_activations.sum(dim=(0, 1))
        stats["max_active_over_zero"] = torch.maximum(stats["max_active_over_zero"], feature_max)
        stats["min_active_over_zero"] = torch.minimum(stats["min_active_over_zero"], feature_min)
        # Update magnitude sums
        stats["activation_sum"] += activation_sum
        stats["activation_squared_sum"] += activation_squared_sum
        
        print(f"[DEBUG] Updated stats for {lang} layer {layer_idx}:")
        print(f"[DEBUG]   num_examples: {old_examples} -> {stats['num_examples']}")
        print(f"[DEBUG]   num_tokens: {old_tokens} -> {stats['num_tokens']}")
        print(f"[DEBUG]   over_zero_token sum: {stats['over_zero_token'].sum().item()}")
        print(f"[DEBUG]   over_zero_example sum: {stats['over_zero_example'].sum().item()}")
        print(f"[DEBUG]   max_active range: {stats['max_active_over_zero'].min().item():.6f} to {stats['max_active_over_zero'].max().item():.6f}")
        print(f"[DEBUG]   activation_sum total: {stats['activation_sum'].sum().item():.6f}")

    def run(self, data_loader, lang):
        """Stream through dataset and aggregate SAE stats."""
        print(f"\n[DEBUG] Starting run for language: {lang}")
        print(f"[DEBUG] Available SAE layers: {list(self.saes.keys())}")
        
        for layer_index, (layer_name, sae_model) in enumerate(self.saes.items()):
            print(f"\n[DEBUG] Processing layer {layer_index} ({layer_name})")
            sae_model.to(self.device)
            
            batch_count = 0
            for batch in tqdm(data_loader, desc=f"{lang} | Layer {layer_name}"):
                print(f"\n[DEBUG] Processing batch {batch_count} for {lang} layer {layer_name}")
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                print(f"[DEBUG] attention_mask shape: {attention_mask.shape}")

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states
                    print(f"[DEBUG] Number of hidden states layers: {len(hidden_states)}")
                    print(f"[DEBUG] Using hidden state at index: {layer_index + 1}")
                    
                    hidden = hidden_states[layer_index + 1]
                    print(f"[DEBUG] hidden shape: {hidden.shape}")
                    print(f"[DEBUG] hidden min: {hidden.min().item():.6f}, max: {hidden.max().item():.6f}")
                    
                    # Check SAE encoding
                    print(f"[DEBUG] SAE model type: {type(sae_model)}")
                    sae_output = sae_model.encode(hidden)
                    print(f"[DEBUG] SAE output type: {type(sae_output)}")
                    print(f"[DEBUG] SAE output attributes: {dir(sae_output)}")
                    # In StreamingExtractor.run(), after sae_output = sae_model.encode(hidden):
                    if hasattr(sae_output, 'feature_acts'):
                        print(f"[DEBUG] feature_acts range: {sae_output.feature_acts.min():.6f} to {sae_output.feature_acts.max():.6f}")
                    if hasattr(sae_output, 'post_acts'):
                        print(f"[DEBUG] post_acts range: {sae_output.post_acts.min():.6f} to {sae_output.post_acts.max():.6f}")
                    sae_latents = sae_output.pre_acts
                    print(f"[DEBUG] sae_latents from pre_acts shape: {sae_latents.shape}")

                self.update_stats(lang, sae_latents, layer_index)
                batch_count += 1

            sae_model.to("cpu")
            torch.cuda.empty_cache()
        
        print(f"\n[DEBUG] Finished processing {lang}")
        print(f"[DEBUG] Final stats summary for {lang}:")
        for i, layer_stats in enumerate(self.lang_to_stats[lang]):
            print(f"[DEBUG]   Layer {i}: {layer_stats['num_examples']} examples, {layer_stats['num_tokens']} tokens")

    def get_stacked_data(self):
        """Convert to original format for sae_lape function."""
        print(f"\n[DEBUG] get_stacked_data called")
        print(f"[DEBUG] Available languages: {list(self.lang_to_stats.keys())}")
        
        from utils.metrics import stack_activations_count
        result = stack_activations_count(self.lang_to_stats, sorted(self.lang_to_stats.keys()))
        
        # Debug the returned data
        (
            num_examples,
            num_tokens, 
            over_zero_token,
            over_zero_example,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
        ) = result
        
        print(f"[DEBUG] Stacked data shapes:")
        print(f"[DEBUG]   num_examples: {num_examples.shape} - {num_examples}")
        print(f"[DEBUG]   num_tokens: {num_tokens.shape} - {num_tokens}")
        print(f"[DEBUG]   over_zero_token: {over_zero_token.shape}")
        print(f"[DEBUG]   over_zero_example: {over_zero_example.shape}")
        print(f"[DEBUG]   global_max_active_over_zero: {global_max_active_over_zero.shape}")
        print(f"[DEBUG]   global_min_active_over_zero: {global_min_active_over_zero.shape}")
        print(f"[DEBUG]   global_avg_active_over_zero: {global_avg_active_over_zero.shape}")
        
        print(f"[DEBUG] over_zero_token sum per language: {over_zero_token.sum(dim=(0,1))}")
        print(f"[DEBUG] over_zero_token nonzero features per language: {(over_zero_token.sum(dim=0) > 0).sum(dim=0)}")
        
        return result

    def get_magnitude_data(self):
        """Convert to format for magnitude ranking."""
        print(f"\n[DEBUG] get_magnitude_data called")
        print(f"[DEBUG] Available languages: {list(self.lang_to_stats.keys())}")
        
        from utils.metrics import stack_magnitude_stats
        result = stack_magnitude_stats(self.lang_to_stats, sorted(self.lang_to_stats.keys()))
        
        return result

    def compute_sae_lape(self, **kwargs):
        """Call original sae_lape function with stacked data."""
        print(f"\n[DEBUG] compute_sae_lape called with kwargs: {kwargs}")
        
        from utils.metrics import sae_lape
        
        stacked_data = self.get_stacked_data()
        (
            num_examples,
            num_tokens, 
            over_zero_token,
            over_zero_example,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
        ) = stacked_data
        
        sorted_lang = sorted(self.lang_to_stats.keys())
        print(f"[DEBUG] Sorted languages: {sorted_lang}")
        
        final_indices, features_info, shared_features = sae_lape(
            num_examples=num_examples,
            num_tokens=num_tokens,
            over_zero_token=over_zero_token,
            over_zero_example=over_zero_example,
            global_max_active_over_zero=global_max_active_over_zero,
            global_min_active_over_zero=global_min_active_over_zero,
            global_avg_active_over_zero=global_avg_active_over_zero,
            sorted_lang=sorted_lang,
            **kwargs
        )
        
        return final_indices, features_info, shared_features

    def compute_magnitude_ranking(self, top=100, top_per_layer=False, apply_filtering=False):
        """Call magnitude ranking function with collected data."""
        print(f"\n[DEBUG] compute_magnitude_ranking called")
        print(f"[DEBUG] Parameters: top={top}, top_per_layer={top_per_layer}, apply_filtering={apply_filtering}")
        
        from utils.metrics import magnitude_ranking
        
        magnitude_data = self.get_magnitude_data()
        (
            num_examples,
            num_tokens,
            activation_sums,
            activation_squared_sums,
            over_zero_token,
            over_zero_example,
        ) = magnitude_data
        
        sorted_lang = sorted(self.lang_to_stats.keys())
        print(f"[DEBUG] Sorted languages: {sorted_lang}")
        
        result = magnitude_ranking(
            num_examples=num_examples,
            num_tokens=num_tokens,
            activation_sums=activation_sums,
            activation_squared_sums=activation_squared_sums,
            over_zero_token=over_zero_token,
            over_zero_example=over_zero_example,
            sorted_lang=sorted_lang,
            top=top,
            top_per_layer=top_per_layer,
            apply_filtering=apply_filtering
        )
        
        return result