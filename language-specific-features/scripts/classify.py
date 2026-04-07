import argparse
import json
import os
from math import inf
from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from bracex import expand
from collector import collect_all_activations, collect_all_sae_features
from const import (
    dataset_choices,
    lang_choices,
    lang_choices_to_iso639_2,
    lang_choices_to_qualified_name,
    layer_to_index,
    model_choices,
    prompt_templates,
    sae_model_choices,
)
from loader import load_dataset_specific
from nnsight import LanguageModel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from utils import (
    TqdmLoggingHandler,
    get_model_dirname,
    get_nested_attr,
    get_project_dir,
    set_deterministic,
)


class DatasetArgs(TypedDict):
    name: str
    start: int
    end: int | float
    split: str


class Args(TypedDict):
    model: str
    dataset: DatasetArgs
    languages: list[str]
    layers: list[str]
    sae_model: str
    local_sae_dir: Path | None
    batch: int
    out_dir: Path
    lape_result_path: Path
    classifier_type: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Classify languages using SAE-LAPE, neurons, or FastText."
    )

    parser.add_argument(
        "dataset",
        help="dataset name",
        type=str,
        choices=dataset_choices,
    )

    parser.add_argument(
        "--model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "--split",
        help="dataset split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--lang",
        help="language(s) to be processed",
        type=str,
        default=[],
        nargs="+",
        choices=lang_choices,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--start",
        help="sample start index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        help="sample end index",
        type=int,
        default=inf,
    )

    parser.add_argument(
        "--sae-model",
        help="sae model name",
        type=str,
        default=None,
        choices=sae_model_choices,
    )

    parser.add_argument(
        "--local-sae-dir",
        help="local directory to load SAE model from",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch",
        help="batch size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--lape-result-path",
        help="path to the lape model",
        type=Path,
    )

    parser.add_argument(
        "--classifier-type",
        help="classifier type",
        type=str,
        default="sae-min-max",
        choices=["sae-min-max", "sae-count", "neuron-count", "fasttext"],
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    return {
        "model": args.model,
        "dataset": {
            "name": args.dataset,
            "start": args.start,
            "end": args.end,
            "split": args.split,
        },
        "languages": args.lang,
        "layers": processed_layers,
        "sae_model": args.sae_model,
        "local_sae_dir": args.local_sae_dir,
        "batch": args.batch,
        "out_dir": args.out_dir,
        "lape_result_path": args.lape_result_path,
        "classifier_type": args.classifier_type,
    }


def calculate_metrics(y_true, y_pred):
    classes = sorted(list(set(y_true) | set(y_pred)))

    accuracy = accuracy_score(y_true, y_pred)

    # Calculate for each class
    precision = precision_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    recall = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    # Calculate macro averages
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "classes": classes,
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_matrix.tolist(),
    }


def sae_count_classifier(args, llm, lape, lang):
    labels = []
    predictions = []
    results = []
    sentences = []

    sorted_lang = lape["sorted_lang"]

    dataset_config = {
        **args["dataset"],
        "lang": lang,
    }

    logger.info(f'Loading Dataset: {dataset_config["name"]} ({dataset_config["lang"]})')

    dataset = load_dataset_specific(
        dataset_config["name"],
        None,
        dataset_config["split"],
        dataset_config["start"],
        dataset_config["end"],
        filter_by_label=dataset_config["lang"],
    )

    prompt_template = prompt_templates[dataset_config["name"]][dataset_config["lang"]]

    all_activations = collect_all_activations(
        llm,
        args["layers"],
        dataset,
        prompt_template,
    )

    all_sae_features = collect_all_sae_features(
        all_activations,
        args["layers"],
        args["model"],
        args["sae_model"],
        args["local_sae_dir"],
        args["batch"],
    )

    del all_activations

    num_examples = dataset_config["end"] - dataset_config["start"]
    result = torch.zeros(num_examples, len(sorted_lang))

    for layer, sae_features_list in tqdm(
        all_sae_features.items(), desc="Processing SAE features", leave=False
    ):
        layer_idx = layer_to_index[layer]
        layer_result = torch.zeros(num_examples, len(sorted_lang))

        for example_idx, sae_features in enumerate(sae_features_list):
            top_indices_flatten = sae_features.top_indices.flatten()

            for lang_idx, lang in enumerate(sorted_lang):
                lang_feature_layer = lape["final_indice"][lang_idx][layer_idx]

                mask = torch.isin(top_indices_flatten, lang_feature_layer)

                if not mask.any():
                    continue

                layer_result[example_idx][lang_idx] = mask.sum()

            result += layer_result

    # Make predictions
    lang_preditions = torch.argmax(result, dim=1)
    result_rounded = [
        [round(elem, ndigits=3) for elem in row] for row in result.tolist()
    ]

    sentences.extend(dataset["sentence"])
    predictions.extend(
        [
            lang_choices_to_iso639_2[sorted_lang[lang_idx]]
            for lang_idx in lang_preditions.tolist()
        ]
    )
    labels.extend(
        [dataset.features["label"].names[label_idx] for label_idx in dataset["label"]]
    )

    results.extend(result_rounded)

    output = pd.DataFrame(
        {
            "sentences": sentences,
            "predictions": predictions,
            "labels": labels,
            f"results ({sorted_lang})": results,
        }
    )

    return output


def sae_min_max_classifier(args, llm, lape, lang):
    labels = []
    predictions = []
    results = []
    sentences = []

    sorted_lang = lape["sorted_lang"]

    dataset_config = {
        **args["dataset"],
        "lang": lang,
    }

    logger.info(f'Loading Dataset: {dataset_config["name"]} ({dataset_config["lang"]})')

    dataset = load_dataset_specific(
        dataset_config["name"],
        None,
        dataset_config["split"],
        dataset_config["start"],
        dataset_config["end"],
        filter_by_label=dataset_config["lang"],
    )

    prompt_template = prompt_templates[dataset_config["name"]][dataset_config["lang"]]

    all_activations = collect_all_activations(
        llm,
        args["layers"],
        dataset,
        prompt_template,
    )

    all_sae_features = collect_all_sae_features(
        all_activations,
        args["layers"],
        args["model"],
        args["sae_model"],
        args["local_sae_dir"],
        args["batch"],
    )

    del all_activations

    num_examples = dataset_config["end"] - dataset_config["start"]
    result = torch.zeros(num_examples, len(sorted_lang))

    for layer, sae_features_list in tqdm(
        all_sae_features.items(), desc="Processing SAE features", leave=False
    ):
        layer_idx = layer_to_index[layer]
        layer_result = torch.zeros(num_examples, len(sorted_lang))

        for example_idx, sae_features in enumerate(sae_features_list):
            top_indices_flatten = sae_features.top_indices.flatten()
            top_acts_flatten = sae_features.top_acts.flatten()

            for lang_idx, lang in enumerate(sorted_lang):
                lang_feature_layer = lape["final_indice"][lang_idx][layer_idx]
                lang_feature_value_max = lape["final_indice_global_max_active"][
                    lang_idx
                ][layer_idx]
                lang_feature_value_min = lape["final_indice_global_min_active"][
                    lang_idx
                ][layer_idx]

                mask = torch.isin(top_indices_flatten, lang_feature_layer)

                if not mask.any():
                    continue

                lang_feature_indices = top_indices_flatten[mask]
                lang_feature_values = top_acts_flatten[mask]

                max_value_map = {
                    idx.item(): val.item()
                    for idx, val in zip(lang_feature_layer, lang_feature_value_max)
                }
                min_value_map = {
                    idx.item(): val.item()
                    for idx, val in zip(lang_feature_layer, lang_feature_value_min)
                }

                lang_feature_values_max = torch.tensor(
                    [max_value_map[idx.item()] for idx in lang_feature_indices]
                )
                lang_feature_values_min = torch.tensor(
                    [min_value_map[idx.item()] for idx in lang_feature_indices]
                )

                lang_feature_values_scaled = (
                    lang_feature_values - lang_feature_values_min
                ) / (lang_feature_values_max - lang_feature_values_min + 1e-6)

                layer_result[example_idx][lang_idx] = lang_feature_values_scaled.sum()

            result += layer_result

    # Make predictions
    lang_preditions = torch.argmax(result, dim=1)
    result_rounded = [
        [round(elem, ndigits=3) for elem in row] for row in result.tolist()
    ]

    sentences.extend(dataset["sentence"])
    predictions.extend(
        [
            lang_choices_to_iso639_2[sorted_lang[lang_idx]]
            for lang_idx in lang_preditions.tolist()
        ]
    )
    labels.extend(
        [dataset.features["label"].names[label_idx] for label_idx in dataset["label"]]
    )

    results.extend(result_rounded)

    output = pd.DataFrame(
        {
            "sentences": sentences,
            "predictions": predictions,
            "labels": labels,
            f"results ({sorted_lang})": results,
        }
    )

    return output


def fasttext_classifier(args, lape, lang):
    import fasttext

    labels = []
    predictions = []
    results = []
    sentences = []

    sorted_lang = lape["sorted_lang"]

    try:
        model_path = args.get("fasttext_model_path", "lid.176.bin")
        lang_model = fasttext.load_model(model_path)
    except:
        logger.info("Downloading FastText language identification model")

        import urllib.request

        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        local_path = Path("lid.176.bin")

        if not local_path.exists():
            urllib.request.urlretrieve(url, local_path)

        lang_model = fasttext.load_model(str(local_path))

    dataset_config = {
        **args["dataset"],
        "lang": lang,
    }

    logger.info(f'Loading Dataset: {dataset_config["name"]} ({dataset_config["lang"]})')

    dataset = load_dataset_specific(
        dataset_config["name"],
        None,
        dataset_config["split"],
        dataset_config["start"],
        dataset_config["end"],
        filter_by_label=dataset_config["lang"],
    )

    for sentence_text in tqdm(dataset["sentence"], desc="Classifying with FastText"):
        predicted_langs, probabilities = lang_model.predict(
            sentence_text, k=len(sorted_lang)
        )

        result = torch.zeros(len(sorted_lang))

        for pred_lang, prob in zip(predicted_langs, probabilities):
            pred_lang_iso639_1 = pred_lang.removeprefix("__label__")
            pred_lang_qualified = lang_choices_to_qualified_name.get(
                pred_lang_iso639_1, None
            )

            if pred_lang_qualified:
                sorted_lang_idx = sorted_lang.index(pred_lang_qualified)
                result[sorted_lang_idx] = prob

        # Add the results
        lang_prediction_idx = torch.argmax(result).item()
        predicted_lang = lang_choices_to_iso639_2[sorted_lang[lang_prediction_idx]]

        sentences.append(sentence_text)
        predictions.append(predicted_lang)
        labels.append(
            dataset.features["label"].names[
                dataset["label"][sentences.index(sentence_text)]
            ]
        )
        results.append([round(x.item(), 3) for x in result])

    output = pd.DataFrame(
        {
            "sentences": sentences,
            "predictions": predictions,
            "labels": labels,
            f"results ({sorted_lang})": results,
        }
    )

    return output


def neuron_count_classifier(args, llm, lape, lang):
    labels = []
    predictions = []
    results = []
    sentences = []

    sorted_lang = lape["sorted_lang"]

    dataset_config = {
        **args["dataset"],
        "lang": lang,
    }

    logger.info(f'Loading Dataset: {dataset_config["name"]} ({dataset_config["lang"]})')

    dataset = load_dataset_specific(
        dataset_config["name"],
        None,
        dataset_config["split"],
        dataset_config["start"],
        dataset_config["end"],
        filter_by_label=dataset_config["lang"],
    )

    prompt_template = prompt_templates[dataset_config["name"]][dataset_config["lang"]]

    num_examples = dataset_config["end"] - dataset_config["start"]
    result = torch.zeros(num_examples, len(sorted_lang))

    layers_modules = {layer: get_nested_attr(llm, layer) for layer in args["layers"]}

    for row_idx, row in tqdm(
        enumerate(dataset), total=len(dataset), desc="Processing Samples", leave=False
    ):
        prompt = prompt_template.format_map(row)

        with llm.trace(prompt):
            for layer in args["layers"]:
                layer_module = layers_modules[layer]
                layer_index = layer_to_index[layer]

                activations = layer_module.output

                for lang_idx, lang in enumerate(sorted_lang):
                    lang_feature_layer = lape["final_indice"][lang_idx][layer_index]
                    neuron_values = activations[:, :, lang_feature_layer]
                    neuron_values_mask = neuron_values > 0

                    result[row_idx][lang_idx] += neuron_values_mask.sum().item()

    # Make predictions
    lang_preditions = torch.argmax(result, dim=1)
    result_rounded = [
        [round(elem, ndigits=3) for elem in row] for row in result.tolist()
    ]

    sentences.extend(dataset["sentence"])
    predictions.extend(
        [
            lang_choices_to_iso639_2[sorted_lang[lang_idx]]
            for lang_idx in lang_preditions.tolist()
        ]
    )
    labels.extend(
        [dataset.features["label"].names[label_idx] for label_idx in dataset["label"]]
    )

    results.extend(result_rounded)

    output = pd.DataFrame(
        {
            "sentences": sentences,
            "predictions": predictions,
            "labels": labels,
            f"results ({sorted_lang})": results,
        }
    )

    return output


@torch.inference_mode()
def main(args: Args):
    logger.info(f'Loading Model: {args["model"]}')

    if args["model"]:
        llm = LanguageModel(args["model"], device_map="auto", dispatch=True)

    lape = torch.load(args["lape_result_path"], weights_only=False)

    for lang in args["languages"]:
        if args["classifier_type"] == "sae-min-max":
            output = sae_min_max_classifier(args, llm, lape, lang)
        elif args["classifier_type"] == "sae-count":
            output = sae_count_classifier(args, llm, lape, lang)
        elif args["classifier_type"] == "fasttext":
            output = fasttext_classifier(args, lape, lang)
        elif args["classifier_type"] == "neuron-count":
            output = neuron_count_classifier(args, llm, lape, lang)

        model_dir = get_model_dirname(args["model"]) if args["model"] else ""
        sae_dir = args["sae_model"] if args["sae_model"] else ""
        output_dir = (
            args["out_dir"]
            / "classification"
            / model_dir
            / sae_dir
            / args["dataset"]["name"]
            / args["classifier_type"]
        )

        os.makedirs(output_dir, exist_ok=True)
        output.to_csv(output_dir / f"predictions_{lang}.csv", index=False)

    results = pd.concat(
        (
            pd.read_csv(output_dir / f"predictions_{lang}.csv")
            for lang in args["languages"]
        ),
        ignore_index=True,
    )

    labels = results["labels"]
    predictions = results["predictions"]
    metrics = calculate_metrics(labels, predictions)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("classify")

    set_deterministic()

    args = parse_args()
    main(args)
