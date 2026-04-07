import argparse
import os
from math import inf
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from bracex import expand
from const import (
    dataset_choices,
    lang_choices,
    lang_choices_to_qualified_name,
    layer_to_index,
    model_choices,
    prompt_templates,
)
from intervention import neurons_intervene_all_token, sae_features_intervene_all_token
from loader import load_dataset_specific
from nnsight import LanguageModel, apply
from tqdm.auto import tqdm
from utils import (
    TqdmLoggingHandler,
    get_device,
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
    out_dir: Path
    out_path: Path
    lape_result_path: Path
    intervention_type: str
    intervention_lang: str
    value: float
    multiplier: float
    layers: list[str]
    lape_value_type: str
    neuron_intervention_method: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Collect activations from a dataset for particular layers and languages"
    )

    parser.add_argument(
        "model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "dataset",
        help="dataset name",
        type=str,
        choices=dataset_choices,
    )

    parser.add_argument(
        "--split",
        help="dataset split",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--local-path",
        help="local path to the dataset",
        type=str,
        default=[],
        nargs="+",
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
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-path",
        help="output path",
        type=Path,
        default=Path("lape.pt"),
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--intervention-type",
        help="intervention type",
        type=str,
        choices=["neuron", "sae-features"],
    )

    parser.add_argument(
        "--intervention-lang",
        help="intervention language",
        type=str,
        choices=lang_choices,
    )

    parser.add_argument(
        "--multiplier",
        help="sae features multiplier",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--value",
        help="value to be used for the neuron intervention",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--neuron-intervention-method",
        help="neuron intervention method",
        type=str,
        default="fixed",
        choices=["fixed", "scaling"],
    )

    parser.add_argument(
        "--lape-result-path",
        help="path to the lape model",
        type=Path,
    )

    parser.add_argument(
        "--lape-value-type",
        help="lape value type",
        type=str,
        default="final_indice_global_max_active",
        choices=[
            "final_indice_global_max_active",
            "final_indice_global_min_active",
            "final_indice_global_avg_active",
        ],
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
        "out_dir": args.out_dir,
        "out_path": args.out_path,
        "layers": processed_layers,
        "intervention_type": args.intervention_type,
        "intervention_lang": args.intervention_lang,
        "lape_result_path": args.lape_result_path,
        "multiplier": args.multiplier,
        "value": args.value,
        "lape_value_type": args.lape_value_type,
        "neuron_intervention_method": args.neuron_intervention_method,
    }


def get_logits(
    llm: LanguageModel,
    prompt: str,
    layers: list[str],
    layers_modules: dict,
    intervention_type: str | None = None,
    intervention_lang_name: str | None = None,
    value: float | None = None,
    multiplier: float | None = None,
    lape: dict | None = None,
    lape_value_type: str = "final_indice_global_max_active",
    neuron_intervention_method: str = "fixed",
):
    with llm.trace(prompt):
        for layer in layers:
            layer_module = layers_modules[layer]
            layer_index = layer_to_index[layer]
            intervention_lang_index = lape["sorted_lang"].index(intervention_lang_name)

            if intervention_type == "neuron":
                layer_module.output = apply(
                    neurons_intervene_all_token,
                    layer_module.output,
                    lape,
                    layer_index,
                    intervention_lang_index,
                    value,
                    neuron_intervention_method,
                    lape_value_type,
                )
            elif intervention_type == "sae-features":
                layer_module.output = apply(
                    sae_features_intervene_all_token,
                    layer_module.output,
                    lape,
                    layer_index,
                    intervention_lang_index,
                    multiplier,
                    lape_value_type,
                )

        logits = llm.lm_head.output.save()

    return logits


def get_logprobs(
    llm: LanguageModel,
    prompt: str,
    layers: list[str],
    layers_modules: dict,
    intervention_type: str | None = None,
    intervention_lang_name: str | None = None,
    value: float | None = None,
    multiplier: float | None = None,
    lape: dict | None = None,
    lape_value_type: str = "final_indice_global_max_active",
    neuron_intervention_method: str = "fixed",
):
    inputs = llm.tokenizer(prompt, return_tensors="pt")
    output_ids = inputs["input_ids"][:, 1:].to(llm.device)

    logits = get_logits(
        llm,
        prompt,
        layers,
        layers_modules,
        intervention_type,
        intervention_lang_name,
        value,
        multiplier,
        lape,
        lape_value_type,
        neuron_intervention_method,
    )
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))

    return logprobs


def compute_perplexity(
    llm: LanguageModel,
    prompt: str,
    layers: list[str],
    layers_modules: dict,
    intervention_type: str | None = None,
    intervention_lang_name: str | None = None,
    value: float | None = None,
    multiplier: float | None = None,
    lape: dict | None = None,
    lape_value_type: str = "final_indice_global_max_active",
    neuron_intervention_method: str = "fixed",
):
    logprobs = get_logprobs(
        llm,
        prompt,
        layers,
        layers_modules,
        intervention_type,
        intervention_lang_name,
        value,
        multiplier,
        lape,
        lape_value_type,
        neuron_intervention_method,
    )
    n_tokens = logprobs.size(1)
    nll_sum = -logprobs.sum()
    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(avg_nll).item()

    return perplexity


@torch.inference_mode()
def main(args: Args):
    logger.info(f'Loading Model: {args["model"]}')

    llm = LanguageModel(args["model"], device_map="auto", dispatch=True)
    layers_modules = {layer: get_nested_attr(llm, layer) for layer in args["layers"]}

    results = {
        lang_choices_to_qualified_name[lang]: {
            "perplexities": [],
            "mean_perplexity": 0,
        }
        for lang in args["languages"]
    }

    device = get_device()

    lape = (
        torch.load(args["lape_result_path"], weights_only=False, map_location=device)
        if args["lape_result_path"]
        else None
    )

    for lang in args["languages"]:
        logger.info(f'Loading Dataset: {args["dataset"]} ({lang})')

        intervention_lang_name = (
            lang_choices_to_qualified_name[args["intervention_lang"]]
            if args["intervention_lang"]
            else None
        )

        normalize_lang = lang_choices_to_qualified_name[lang]

        dataset = load_dataset_specific(
            args["dataset"]["name"],
            lang,
            args["dataset"]["split"],
            args["dataset"]["start"],
            args["dataset"]["end"],
        )

        prompt_template = prompt_templates[args["dataset"]["name"]][lang]

        for row in tqdm(dataset, desc="Processing Samples"):
            prompt = prompt_template.format_map(row)
            ppl = compute_perplexity(
                llm,
                prompt,
                args["layers"],
                layers_modules,
                args["intervention_type"],
                intervention_lang_name,
                args["value"],
                args["multiplier"],
                lape,
                args["lape_value_type"],
                args["neuron_intervention_method"],
            )
            results[normalize_lang]["perplexities"].append(ppl)

        results[normalize_lang]["mean_perplexity"] = np.mean(
            results[normalize_lang]["perplexities"]
        ).item()

    model_dir = get_model_dirname(args["model"])
    output_path = (
        args["out_dir"]
        / "ppl"
        / model_dir
        / args["dataset"]["name"]
        / args["out_path"]
    )
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(results, output_path)


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("activations")

    set_deterministic()

    args = parse_args()

    logger.info(f"Arguments: {args}")

    main(args)
