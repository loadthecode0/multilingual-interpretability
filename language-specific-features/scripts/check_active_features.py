import argparse
import json
import os
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from collector import collect_all_activations, collect_all_sae_features
from const import (
    lang_choices_to_iso639_2,
    layer_to_index,
    model_choices,
    sae_model_choices,
)
from datasets import Dataset
from loader import load_all_interpretations
from nnsight import LanguageModel
from tqdm.auto import tqdm
from utils import (
    TqdmLoggingHandler,
    get_model_dirname,
    get_project_dir,
    set_deterministic,
)


class Args(TypedDict):
    model: str
    layers: list[str]
    sae_model: str
    local_sae_dir: Path | None
    batch: int
    out_dir: Path
    lape_result_path: Path
    classifier_type: str
    text: list[str]
    interpretation_folder: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Check active language-specific features in a model given input text(s) using SAE."
    )

    parser.add_argument(
        "--model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
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
        default="min-max",
        choices=["min-max", "count", "fasttext"],
    )

    parser.add_argument(
        "--text",
        help="text(s) to be processed",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--interpretation-folder",
        help="folder to save the interpretation results",
        type=Path,
        default=get_project_dir() / "interpret_sae_features" / "explanations",
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    return {
        "model": args.model,
        "layers": processed_layers,
        "sae_model": args.sae_model,
        "local_sae_dir": args.local_sae_dir,
        "batch": args.batch,
        "out_dir": args.out_dir,
        "lape_result_path": args.lape_result_path,
        "classifier_type": args.classifier_type,
        "text": args.text,
        "interpretation_folder": args.interpretation_folder,
    }


def count_classifier(args, llm, lape, text):
    predictions = []
    sentences = []
    results = []

    sorted_lang = lape["sorted_lang"]

    dataset = Dataset.from_dict({"text": text})
    prompt_template = "{text}"

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

    num_examples = len(dataset)

    result = torch.zeros(num_examples, len(sorted_lang))
    features = [{} for _ in range(num_examples)]

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

                lang_feature_indices = top_indices_flatten[mask]

                token_indices = (
                    torch.isin(sae_features.top_indices, lang_feature_indices)
                    .any(dim=-1)
                    .squeeze()
                )

                if layer not in features[example_idx]:
                    features[example_idx][layer] = {}

                features[example_idx][layer][lang] = {
                    feature_index: {
                        "interpretation": None,
                        "tokens": token_indices.nonzero(as_tuple=True)[0].tolist(),
                    }
                    for feature_index in lang_feature_indices.unique().tolist()
                }

                layer_result[example_idx][lang_idx] = mask.sum()

            result += layer_result

    # Make predictions
    lang_preditions = torch.argmax(result, dim=1)
    result_rounded = [
        [round(elem, ndigits=3) for elem in row] for row in result.tolist()
    ]

    sentences.extend(text)
    predictions.extend(
        [
            lang_choices_to_iso639_2[sorted_lang[lang_idx]]
            for lang_idx in lang_preditions.tolist()
        ]
    )

    results.extend(result_rounded)

    return sentences, predictions, features


def extract_results(
    args, llm, sentences, predictions, features, interpretations
):
    results = []

    for i in range(len(args["text"])):
        result = {
            "sentences": sentences[i],
            "predictions": predictions[i],
            "features": features[i],
        }

        tokenized = llm.tokenizer(sentences[i], return_tensors="pt")
        tokenized_input_ids = tokenized["input_ids"].squeeze(0)
        tokenized_str = [
            llm.tokenizer.decode(input_id) for input_id in tokenized_input_ids.tolist()
        ]

        for layer_index, lang_features in result["features"].items():
            for _, indices in lang_features.items():
                for feature_index in indices:
                    indices[feature_index] = {
                        "interpretation": interpretations[layer_index][feature_index],
                        "tokens": [
                            tokenized_str[token_id]
                            for token_id in indices[feature_index]["tokens"]
                        ],
                    }

        results.append(result)

    return results


@torch.inference_mode()
def main(args: Args):
    logger.info(f'Loading Model: {args["model"]}')

    if args["model"]:
        llm = LanguageModel(args["model"], device_map="auto", dispatch=True)

    lape = torch.load(args["lape_result_path"], weights_only=False)

    if args["classifier_type"] == "count":
        sentences, predictions, features = count_classifier(
            args, llm, lape, args["text"]
        )

    model_dir = get_model_dirname(args["model"])
    output_dir = (
        args["out_dir"]
        / "active_features"
        / model_dir
        / args["sae_model"]
        / args["classifier_type"]
    )

    interpretations = load_all_interpretations(args["interpretation_folder"])

    results = extract_results(
        args, llm, sentences, predictions, features, interpretations
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / "result.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("classify")

    set_deterministic()

    args = parse_args()
    main(args)
