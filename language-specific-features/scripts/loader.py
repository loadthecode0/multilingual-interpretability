import inspect
import os
import re
import sys
from ast import literal_eval
from collections import defaultdict
import json
from functools import reduce
from math import isinf
from pathlib import Path
from typing import Literal

# Add parent directory to sys.path to find utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import paths

import pandas as pd
import torch
from bracex import expand
from const import (
    lang_choices_to_flores,
    lang_choices_to_iso639_1,
    lang_choices_to_iso639_2,
    lang_choices_to_qualified_name,
    layer_to_index,
    prompt_templates,
    sae_model_layer_to_hookpoint,
)
from datasets import Dataset, get_dataset_config_names
from datasets import load_dataset as hg_load_dataset
from sparsify import Sae
from sparsify.sparse_coder import EncoderOutput
from tqdm.auto import tqdm


def load_env(name: str, env: Literal["local", "colab", "kaggle"]):
    if env == "local":
        from dotenv import load_dotenv

        load_dotenv()
        env_var = os.getenv(name)
    elif env == "colab":
        from google.colab import userdata

        env_var = userdata.get(name)
    elif env == "kaggle":
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        env_var = user_secrets.get_secret(name)

    return env_var


def load_dataset(
    dataset_name: str,
    config_names: list[str] | Literal["all"],
    split: str,
) -> dict[str, Dataset]:
    if config_names == "all":
        config_names = get_dataset_config_names(dataset_name)

    dataset = {
        config_name: hg_load_dataset(dataset_name, config_name, split=split)
        for config_name in config_names
    }

    return dataset


def load_dataset_specific(
    dataset_name: str,
    config_name: str | None,
    split: str,
    start: int,
    end: int | float,
    filter_by_label: str | None = None,
) -> Dataset:
    end = -1 if isinf(end) else end

    # Handle Dakshina dataset
    if dataset_name == "dakshina":
        base_dir = Path("data/multilingual_datasets/dakshina")
        file_path = base_dir / config_name / "romanized" / f"{config_name}.romanized.rejoined.tsv"
        
        if not file_path.exists():
            # Fallback to absolute path for now if relative fails
            base_dir = Path(paths.DAKSHINA_DATASET_V1)
            file_path = base_dir / config_name / "romanized" / f"{config_name}.romanized.rejoined.tsv"
            if not file_path.exists():
                raise FileNotFoundError(f"Dakshina dataset file not found: {file_path}")
        
        import pandas as pd
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["native", "romanized"],
            engine="python",         # more forgiving than C parser
            quoting=3,               # ignore quotes
            on_bad_lines="skip"      # skip malformed rows
        )
        
        # Select the specified range
        df = df.iloc[start:end]
        dataset = Dataset.from_pandas(df)
        
        print(f"[DAKSHINA DEBUG] Loaded {len(dataset)} examples for {config_name}")
        if len(dataset) > 0:
            print(f"[DAKSHINA DEBUG] Example: {dataset[0]}")
        
        return dataset

    # Handle combined ICU + Dakshina dataset
    if dataset_name == "icu_dakshina":
        # indic_langs = {"hi", "mr", "bn", "ur"}
        indic_langs = {}
        print(config_name, dataset_name)
        if config_name in indic_langs:
            base_dir = Path("data/multilingual_datasets/dakshina")
            file_path = base_dir / config_name / "romanized" / f"{config_name}.romanized.rejoined.tsv"
            
            if not file_path.exists():
                base_dir = Path(paths.DAKSHINA_DATASET_V1)
                file_path = base_dir / config_name / "romanized" / f"{config_name}.romanized.rejoined.tsv"
                if not file_path.exists():
                    raise FileNotFoundError(f"Dakshina dataset file not found: {file_path}")
            
            import pandas as pd
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=["native", "romanized"],
                engine="python",
                quoting=3,
                on_bad_lines="skip",
            )
            df = df.iloc[start:end]
            print(len(df))
            return Dataset.from_pandas(df)
        else:
            # Load from ICU-curated FLORES-PLUS romanized JSONL
            flores_lang = lang_choices_to_flores[config_name]
            base_dir = Path(paths.FLORES_ROMANIZED_DIR)
            file_path = base_dir / f"{flores_lang}.{split}.romanized.jsonl"
            if not file_path.exists():
                raise FileNotFoundError(f"ICU FLORES jsonl not found: {file_path}")
            import pandas as pd
            records = []
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    native = obj.get("text_original", "")
                    romanized = obj.get("text_romanized", "")
                    records.append({"native": native, "romanized": romanized})
            df = pd.DataFrame.from_records(records)
            df = df.iloc[start:end]
            print(len(df))
            return Dataset.from_pandas(df)

    if filter_by_label:
        dataset = hg_load_dataset(
            dataset_name,
            config_name,
            split=split,
            trust_remote_code=True,
        )

        labels = dataset.features["label"].names
        label_index = labels.index(filter_by_label)
        dataset = dataset.filter(lambda row: row["label"] == label_index)
        dataset = dataset.select(range(start, end))
    else:
        dataset = hg_load_dataset(
            dataset_name,
            config_name,
            split=f"{split}[{start}:{end}]",
            trust_remote_code=True,
        )

    return dataset


def load_dataset_specific_rows(
    dataset_name: str,
    config_name: str,
    split: str,
    row_ids: list[int],
) -> Dataset:
    dataset = hg_load_dataset(
        dataset_name,
        config_name,
        split=split,
    )

    return dataset.select(row_ids)


def load_activations(input_dir: Path, layer: str, logger=None):
    torch.serialization.add_safe_globals([EncoderOutput])

    layer_files = sorted(list(input_dir.glob(f"{layer}*.pt")), key=extract_range)

    if logger:
        logger.info(
            f"Loading activations from {input_dir}: {[layer_file.name for layer_file in layer_files]}"
        )

        logger.info(layer_files)

    activations = []

    for layer_file in layer_files:
        activations.extend(torch.load(layer_file, weights_only=False))

    return activations


def extract_range(file_path: Path):
    m = re.search(r"(\d+)-(\d+)", file_path.name)

    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return start, end

    return (0, 0)


def load_sae(
    model_name: str, sae_model_name: str, layer: str, local_sae_dir: Path | None = None
):
    if sae_model_name.startswith("EleutherAI/"):
        # if sae_model_name.endswith("/sae"):
        if local_sae_dir:
            file_dir = (
                local_sae_dir
                / sae_model_layer_to_hookpoint["meta-llama/Llama-3.2-1B"][sae_model_name][layer]
            )
            sae = Sae.load_from_disk(file_dir)
        else:
            sae = Sae.load_from_hub(
                sae_model_name,
                sae_model_layer_to_hookpoint[model_name][sae_model_name][layer],
            )
    else:
        from sae_lens import SAE

        sae_class_file = inspect.getfile(SAE)
        print(f"[SAE DEBUG] SAE class defined in: {sae_class_file}")

        layer_index = layer.split(".")[-2]
        sae_layer = f"layer_{layer_index}"
        root_dir = f"{sae_layer}/width_65k/canonical"
        print(f"[SAE DEBUG] Loading SAE from: {root_dir}")
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(
            release=sae_model_name,
            sae_id=root_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


    return sae


def load_layer_to_summary(
    input_dir: Path, layers: list[str], langs: list[str]
) -> dict[str, pd.DataFrame]:
    layer_to_statistics = {}

    for layer in layers:
        layer_to_statistic = None

        for lang in langs:
            file_path = input_dir / layer / f"{lang}.csv"
            df_lang_layer = pd.read_csv(file_path)
            layer_to_statistic = pd.concat(
                [layer_to_statistic, df_lang_layer], ignore_index=True
            )

        layer_to_statistics[layer] = layer_to_statistic

    return layer_to_statistics


def load_lang_to_dataset_token_activations(
    input_dir: Path, layer: str, langs: list[str], feature_indicies: list[int] = None
) -> dict[str, pd.DataFrame]:
    lang_to_dataset_token_activations = {}

    for lang in langs:
        df_lang_layer = pd.read_csv(input_dir / layer / f"{lang}.csv")

        if feature_indicies:
            df_lang_layer = df_lang_layer[df_lang_layer["index"].isin(feature_indicies)]

        lang_to_dataset_token_activations[lang] = df_lang_layer

    return lang_to_dataset_token_activations


def load_lang_to_dataset_token_activations_aggregate(
    input_dir: Path, layers: list[str], langs: list[str]
) -> dict[str, pd.DataFrame]:
    combined_df = None

    for lang in langs:
        for layer in layers:
            df_lang_layer = pd.read_csv(input_dir / layer / f"{lang}.csv")

            df_lang_layer.rename(
                {
                    "index": "sae_feature_number",
                    "count": "token_count",
                }
            )

            df_lang_layer["layer_number"] = layer_to_index[layer]
            df_lang_layer["language"] = lang_choices_to_qualified_name[lang]

            combined_df = pd.concat([combined_df, df_lang_layer], ignore_index=True)

    return combined_df


def load_from_dataset_configs(
    dataset_configs: str, dataset_start: int, dataset_end: int, text_column: str, shuffle_words: bool = False, romanized: bool = False
):
    processed_dataset_configs = []

    for dataset_config in dataset_configs:
        processed_dataset_configs.extend(expand(dataset_config))

    lang_merged_dataset = defaultdict(list)
    example_count = 0

    for dataset_config in tqdm(processed_dataset_configs, desc="Loading datasets"):
        dataset_name, config_name, split_name = dataset_config.split(":")
        config_name = None if config_name == "" else config_name

        tqdm.write(f"Dataset: {dataset_name}, Config: {config_name}, Split: {split_name}")

        dataset = load_dataset_specific(
            dataset_name, config_name, split_name, dataset_start, dataset_end
        )

        prompt_template = prompt_templates[dataset_name][config_name]
        
        tqdm.write(f"Number of samples: {len(dataset)}")
        for row in tqdm(dataset, desc="Processing Samples", leave=False):
            # Choose prompt based on romanized flag
            tqdm.write(f"Row: {row}")
            if romanized and dataset_name in {"dakshina", "icu_dakshina"}:
                prompt = row["romanized"]  # Use romanized field directly
            else:
                prompt = prompt_template.format_map(row)
                if i < 3:
                    tqdm.write(f"Prompt: {prompt}")
                    i+=1
            
            # Shuffle words if requested
            if shuffle_words:
                import random
                original_prompt = prompt
                words = prompt.split()
                random.shuffle(words)
                prompt = " ".join(words)
                # Print first few examples to show shuffling
                if example_count < 3:
                    print(f"[SHUFFLE DEBUG] Example {example_count + 1}:")
                    print(f"  Original: {original_prompt}")
                    print(f"  Shuffled: {prompt}")
                    print()
                example_count += 1
            
            lang_merged_dataset[lang_choices_to_qualified_name[config_name]].append(
                prompt
            )

            tqdm.write(f"Length of prompt: {len(prompt)}")

    for lang, lang_dataset in lang_merged_dataset.items():
        print(f"{lang}: {len(lang_dataset)} samples")

    merged_dataset = []

    for lang_dataset in lang_merged_dataset.values():
        merged_dataset.extend(lang_dataset)

    merged_dataset = Dataset.from_dict({text_column: merged_dataset})

    return merged_dataset


def load_all_interpretations(
    interpretation_folder: Path,
) -> defaultdict[defaultdict[str]]:
    interpretations = defaultdict(dict)

    for file_path in interpretation_folder.iterdir():
        layer, feature_index = file_path.stem.split("_latent")

        with file_path.open("r", encoding="utf-8") as file:
            interpretations[f"model.{layer}"][int(feature_index)] = file.read()

    return interpretations


def load_task_df(
    lape_result,
    lang,
    lang_index,
    layers,
    task_configs,
):
    result = {
        "xnli": None,
        "paws-x": None,
        "flores": None,
    }

    for layer in tqdm(layers, desc="Processing layers", leave=False):
        layer_index = layer_to_index[layer]
        lang_final_indices = lape_result["final_indice"][lang_index][
            layer_index
        ].tolist()

        if len(lang_final_indices) == 0:
            continue

        layer = layers[layer_index]

        lang_to_dataset_token_activations_xnli = load_lang_to_dataset_token_activations(
            task_configs["xnli"]["path"],
            layer,
            task_configs["xnli"]["config"]["languages"],
            lang_final_indices,
        )

        if lang_choices_to_iso639_1[lang] in lang_to_dataset_token_activations_xnli:
            xnli_df = lang_to_dataset_token_activations_xnli[
                lang_choices_to_iso639_1[lang]
            ]
            xnli_df["layer"] = layer

            xnli_df["dataset_row_id_token_id_act_val"] = xnli_df[
                "dataset_row_id_token_id_act_val"
            ].apply(literal_eval)

            result["xnli"] = pd.concat(
                [
                    result["xnli"],
                    xnli_df,
                ]
            )

        lang_to_dataset_token_activations_pawsx = (
            load_lang_to_dataset_token_activations(
                task_configs["paws-x"]["path"],
                layer,
                task_configs["paws-x"]["config"]["languages"],
                lang_final_indices,
            )
        )

        if lang_choices_to_iso639_2[lang] in lang_to_dataset_token_activations_pawsx:
            pawsx_df = lang_to_dataset_token_activations_pawsx[
                lang_choices_to_iso639_2[lang]
            ]
            pawsx_df["layer"] = layer
            pawsx_df["dataset_row_id_token_id_act_val"] = pawsx_df[
                "dataset_row_id_token_id_act_val"
            ].apply(literal_eval)

            result["paws-x"] = pd.concat(
                [
                    result["paws-x"],
                    pawsx_df,
                ]
            )

        lang_to_dataset_token_activations_flores = (
            load_lang_to_dataset_token_activations(
                task_configs["flores"]["path"],
                layer,
                task_configs["flores"]["config"]["languages"],
                lang_final_indices,
            )
        )

        if lang_choices_to_flores[lang] in lang_to_dataset_token_activations_flores:
            flores_df = lang_to_dataset_token_activations_flores[
                lang_choices_to_flores[lang]
            ]
            flores_df["layer"] = layer
            flores_df["dataset_row_id_token_id_act_val"] = flores_df[
                "dataset_row_id_token_id_act_val"
            ].apply(literal_eval)
            result["flores"] = pd.concat(
                [
                    result["flores"],
                    flores_df,
                ]
            )

    combined_df = []

    for task, df in result.items():
        if df is not None:
            df.rename(
                columns={
                    "count": f"{task}_count",
                    "dataset_row_id_token_id_act_val": f"{task}_dataset_row_id_token_id_act_val",
                },
                inplace=True,
            )

            combined_df.append(df)

    final_result = reduce(
        lambda left, right: pd.merge(left, right, on=["index", "layer"]), combined_df
    )

    return final_result


def load_sae_features_info_df(lape_result, layers, metrics):
    sae_feature_info_records = []

    sorted_lang = lape_result["sorted_lang"]

    for lang in tqdm(sorted_lang, desc="Processing languages"):
        lang_index = sorted_lang.index(lang)

        for layer in layers:
            layer_index = layer_to_index[layer]
            lang_final_indices = lape_result["final_indice"][lang_index][
                layer_index
            ].tolist()

            if len(lang_final_indices) == 0:
                continue

            layer = layers[layer_index]
            selected_probs = lape_result["features_info"][lang]["selected_probs"]
            entropies = lape_result["features_info"][lang]["entropies"]

            for feature_index in lang_final_indices:
                arg_index = lape_result["features_info"][lang]["indicies"].index(
                    (layer_index, feature_index)
                )

                feature_info = {
                    "feature_index": feature_index,
                    "layer": layer,
                    "lang": lang,
                    "selected_prob": round(selected_probs[arg_index].item(), ndigits=3),
                    "entropy": round(entropies[arg_index].item(), ndigits=3),
                    "metrics": metrics[layer][feature_index],
                }

                # Extract metrics for each score type
                for metric in feature_info["metrics"]:
                    record = {
                        "entropy": feature_info["entropy"],
                        "selected_prob": feature_info["selected_prob"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1_score": metric["f1_score"],
                        "accuracy": metric["accuracy"],
                        "score_type": metric["score_type"],
                        "layer": feature_info["layer"],
                        "lang": feature_info["lang"],
                        "feature_index": feature_info["feature_index"],
                    }
                    sae_feature_info_records.append(record)

    sae_features_info = pd.DataFrame(sae_feature_info_records)

    return sae_features_info


def load_lang_to_sae_features_info(
    lape_result, layers, interpretations, metrics, extra: bool = False
):
    sorted_lang = lape_result["sorted_lang"]

    lang_to_sae_features_info = {
        lang: {layer: {} for layer in layers} for lang in sorted_lang
    }

    for lang in tqdm(sorted_lang, desc="Processing languages"):
        lang_index = sorted_lang.index(lang)

        for layer in layers:
            layer_index = layer_to_index[layer]
            lang_final_indices = lape_result["final_indice"][lang_index][
                layer_index
            ].tolist()

            if len(lang_final_indices) == 0:
                continue

            layer = layers[layer_index]
            selected_probs = lape_result["features_info"][lang]["selected_probs"]
            entropies = lape_result["features_info"][lang]["entropies"]

            for feature_index in lang_final_indices:
                arg_index = lape_result["features_info"][lang]["indicies"].index(
                    (layer_index, feature_index)
                )

                feature_info = {
                    "Layer": layer,
                    "Lang": lang,
                    "Feature ID": feature_index,
                    "Interpretation": interpretations[layer][feature_index],
                }

                if extra:
                    feature_info.update(
                        {
                            "Selected Prob": round(
                                selected_probs[arg_index].item(), ndigits=3
                            ),
                            "Entropy": round(entropies[arg_index].item(), ndigits=3),
                            "Metrics": metrics[layer][feature_index],
                        }
                    )

                lang_to_sae_features_info[lang][layer][feature_index] = feature_info

    return lang_to_sae_features_info
