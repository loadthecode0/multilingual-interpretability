import pandas as pd
import os
import glob
import argparse
from scipy import stats
import numpy as np

def get_metrics_df(input_root, model, category_type, lang):
    # Try different ablation types as some experiments use zero and others use mean
    ablation_types = ["ablation_zero_degreeall", "ablation_mean_degreeall"]
    for ablation in ablation_types:
        path = os.path.join(input_root, model, "raw", category_type, ablation, "layers_all", lang, "metrics.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def process_lang(lang, exp_type, model, input_root, mappings):
    if lang not in mappings[exp_type]:
        return None

    lang_mapping = mappings[exp_type][lang]
    results = []
    metrics = ['kl', 'ppl_clean', 'ppl_patch', 'ppl_ratio', 'ppl_delta']

    for target, control in lang_mapping.items():
        df_target = get_metrics_df(input_root, model, target, lang)
        df_control = get_metrics_df(input_root, model, control, lang)

        if df_target is None or df_control is None:
            print(f"Warning: Could not find results for {target} or {control} in {lang}")
            continue

        # Add ppl_delta if both patch and clean are present
        for df in [df_target, df_control]:
            if 'ppl_patch' in df.columns and 'ppl_clean' in df.columns:
                df['ppl_delta'] = df['ppl_patch'] - df['ppl_clean']

        # Ensure they have the same length and indices for paired t-test
        df_target = df_target.sort_values('idx')
        df_control = df_control.sort_values('idx')
        
        common_indices = np.intersect1d(df_target['idx'], df_control['idx'])
        df_target = df_target[df_target['idx'].isin(common_indices)]
        df_control = df_control[df_control['idx'].isin(common_indices)]

        res = {
            "lang": lang,
            "category": target.replace(f"{exp_type}_", ""),
            "control": control.replace("shuffling_", ""),
            "num_examples": len(common_indices)
        }

        for m in metrics:
            if m in df_target.columns and m in df_control.columns:
                target_vals = df_target[m].values
                control_vals = df_control[m].values
                
                mean_t = np.mean(target_vals)
                mean_c = np.mean(control_vals)
                
                t_stat, p_val = stats.ttest_rel(target_vals, control_vals)
                
                res[f"{m}_mean_target"] = mean_t
                res[f"{m}_mean_control"] = mean_c
                res[f"{m}_diff"] = mean_t - mean_c
                res[f"{m}_p_val"] = p_val
                res[f"{m}_significant"] = p_val < 0.05
                res[f"{m}_direction"] = "target > control" if mean_t > mean_c else "target < control"

        results.append(res)
    return results

def main():
    parser = argparse.ArgumentParser(description="Summarize causal intervention results with statistical testing.")
    parser.add_argument("--lang", type=str, default="all", help="Language code (e.g., en, hi, fr, zh) or 'all'")
    parser.add_argument("--model", type=str, default="Llama-3.2-1B", help="Model name")
    parser.add_argument("--experiment", type=str, default="shuffling", choices=["shuffling", "romanization"], help="Experiment type")
    parser.add_argument("--input_root", type=str, default="causal_results_neuron_lists_100ex", help="Input root directory")
    args = parser.parse_args()

    model = args.model
    input_root = args.input_root
    exp_type = args.experiment

    MAPPINGS = {
        "shuffling": {
            "en": {"shuffling_overlap": "shuffling_random_2", "shuffling_only_normal": "shuffling_random_3"},
            "hi": {"shuffling_overlap": "shuffling_random_12", "shuffling_only_normal": "shuffling_random_4"},
            "fr": {"shuffling_overlap": "shuffling_random_10", "shuffling_only_normal": "shuffling_random_5"},
            "zh": {"shuffling_overlap": "shuffling_random_6", "shuffling_only_normal": "shuffling_random_2"},
            "ru": {"shuffling_overlap": "shuffling_random_7", "shuffling_only_normal": "shuffling_random_2"}
        },
        "romanization": {
            "en": {"romanization_overlap": "romanization_random_1", "romanization_only_native": "romanization_random_6"},
            "hi": {"romanization_overlap": "romanization_random_3", "romanization_only_native": "romanization_random_10"}
        }
    }

    langs_to_process = [args.lang] if args.lang != "all" else list(MAPPINGS[exp_type].keys())
    
    all_results = []
    for lang in langs_to_process:
        res = process_lang(lang, exp_type, model, input_root, MAPPINGS)
        if res:
            all_results.extend(res)

    if not all_results:
        print(f"No results found.")
        return

    summary_df = pd.DataFrame(all_results)
    output_dir = os.path.join(input_root, "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{args.lang}_{model}_{exp_type}_comparison_summary.csv")
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nComparison Summary for experiment: {exp_type}, language: {args.lang}, model: {model}")
    print(f"Results saved to: {output_file}\n")
    
    # Display a condensed version focusing on ppl_ratio and ppl_delta
    display_cols = ["lang", "category", "control"]
    for m in ["ppl_ratio", "ppl_delta"]:
        if f"{m}_diff" in summary_df.columns:
            display_cols.extend([f"{m}_diff", f"{m}_p_val"])
    
    print(summary_df[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
