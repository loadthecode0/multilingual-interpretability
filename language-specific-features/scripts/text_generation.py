import argparse
import os
from pathlib import Path
from typing import TypedDict

import nnsight
import pandas as pd
import torch
from bracex import expand
from const import (
    lang_choices,
    lang_choices_to_qualified_name,
    layer_to_index,
    model_choices,
)
from intervention import (
    sae_features_intervene_all_token,
    sae_features_intervene_last_token,
)
from nnsight import LanguageModel, apply
from tqdm.auto import tqdm
from transformers import set_seed
from utils import TqdmLoggingHandler, get_device, get_nested_attr, get_project_dir


class Args(TypedDict):
    model: str
    layers: list[str]
    in_dir: Path
    in_path: Path
    out_dir: Path
    out_path: Path
    multiplier: float
    max_new_token: int
    times: int
    seed: int
    lang: list[str]
    prompt: str
    last_token_only: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Visualize sae features from a dataset for particular layers and languages."
    )

    parser.add_argument(
        "model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be intervened. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--in-dir",
        help="input directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--in-path",
        help="input path",
        type=Path,
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
    )

    parser.add_argument(
        "--multiplier",
        help="sae features multiplier",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--max-new-token",
        help="Maximum number of new tokens to generate",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--times",
        help="How many times to generate text",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        help="Seed for reproducibility",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lang",
        help="language(s) to intervene.",
        type=str,
        default=[],
        nargs="+",
        choices=lang_choices,
    )

    parser.add_argument(
        "--prompt",
        help="Prompt for the text generation",
        type=str,
        default="",
    )

    parser.add_argument(
        "--last-token-only",
        help="Intervene on the last token only",
        action="store_true",
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    return {
        "model": args.model,
        "layers": processed_layers,
        "in_dir": args.in_dir,
        "in_path": args.in_path,
        "out_dir": args.out_dir,
        "out_path": args.out_path,
        "multiplier": args.multiplier,
        "max_new_token": args.max_new_token,
        "times": args.times,
        "seed": args.seed,
        "lang": args.lang,
        "prompt": args.prompt,
        "last_token_only": args.last_token_only,
    }


@torch.inference_mode()
def generate_text(
    llm: LanguageModel,
    sae_features_specific,
    lang_index: int,
    args: dict,
):
    layers = args["layers"]
    layers_modules = {layer: get_nested_attr(llm, layer) for layer in layers}

    if args["last_token_only"]:
        decoded_prompt, decoded_answer, logits = generate_text_last_token(
            llm,
            sae_features_specific,
            lang_index,
            args["prompt"],
            args["max_new_token"],
            args["multiplier"],
            layers,
            layers_modules,
        )
    else:
        decoded_prompt, decoded_answer, logits = generate_text_all_token(
            llm,
            sae_features_specific,
            lang_index,
            args["prompt"],
            args["max_new_token"],
            args["multiplier"],
            layers,
            layers_modules,
        )

    return decoded_prompt, decoded_answer, logits


def generate_text_last_token(
    llm,
    sae_features_specific,
    lang_index,
    prompt,
    max_new_tokens,
    multiplier,
    layers,
    layers_modules,
):

    with llm.generate(
        prompt, max_new_tokens=max_new_tokens, pad_token_id=llm.tokenizer.eos_token_id
    ):
        logits = nnsight.list().save()

        for _ in range(max_new_tokens):
            # Apply the intervention
            for layer in layers:
                layer_module = layers_modules[layer]
                layer_index = layer_to_index[layer]

                layer_module.output = apply(
                    sae_features_intervene_last_token,
                    layer_module.output,
                    sae_features_specific,
                    layer_index,
                    lang_index,
                    multiplier,
                    "final_indice_global_max_active",
                )

                layer_module.next()

            lm_head = llm.lm_head
            logits.append(lm_head.output)
            lm_head.next()

        # Save the generated text
        out = llm.generator.output.save()

    decoded_prompt = llm.tokenizer.decode(out[0][:-max_new_tokens].cpu())
    decoded_answer = llm.tokenizer.decode(out[0][-max_new_tokens:].cpu())

    return decoded_prompt, decoded_answer, logits


def generate_text_all_token(
    llm,
    sae_features_specific,
    lang_index,
    prompt,
    max_new_tokens,
    multiplier,
    layers,
    layers_modules,
):
    inital_prompt = prompt

    logits = []

    for _ in range(max_new_tokens):
        with llm.generate(
            prompt, max_new_tokens=1, pad_token_id=llm.tokenizer.eos_token_id
        ):
            current_logits = nnsight.list().save()

            # Apply the intervention
            for layer in layers:
                layer_module = layers_modules[layer]
                layer_index = layer_to_index[layer]

                layer_module.output = apply(
                    sae_features_intervene_all_token,
                    layer_module.output,
                    sae_features_specific,
                    layer_index,
                    lang_index,
                    multiplier,
                    "final_indice_global_max_active",
                )

            lm_head = llm.lm_head
            current_logits.append(lm_head.output)
            lm_head.next()

            # Save the generated text
            out = llm.generator.output.save()

        logits.extend(current_logits)
        prompt = llm.tokenizer.decode(out[0][1:].cpu())

        if out[0][-1] == llm.tokenizer.eos_token_id:
            break

    decoded_prompt = inital_prompt
    decoded_answer = prompt[len(inital_prompt) :]

    return decoded_prompt, decoded_answer, logits


def main(args: Args):
    logger.info(f'Loading Model: {args["model"]}')

    llm = LanguageModel(args["model"], device_map="auto", dispatch=True)

    input_dir = args["in_dir"]
    file_path = input_dir / args["in_path"]

    device = get_device()
    lape = torch.load(file_path, weights_only=False, map_location=device)
    sorted_lang = lape["sorted_lang"]

    # Generate texts for each language
    for lang in args["lang"]:
        generation_result = {
            "seed": [],
            "generated_text": [],
        }
        lang_name = lang_choices_to_qualified_name[lang]
        lang_index = sorted_lang.index(lang_name)

        for index in tqdm(
            range(args["times"]), desc=f"Generating texts for {lang_name}"
        ):
            seed = args["seed"] + index
            set_seed(seed)

            prompt, generated_text, logits = generate_text(
                llm,
                lape,
                lang_index,
                args,
            )

            logger.info(f"Result-{index}:\n{prompt}{generated_text}")

            generation_result["seed"].append(seed)
            generation_result["generated_text"].append(generated_text)

        output_dir = args["out_dir"] / args["out_path"] / lang
        layers = [layer_to_index[layer] for layer in args["layers"]]

        file_path = (
            output_dir
            / f"seed_{args['seed']}_mult_{args['multiplier']}_token_{args['max_new_token']}_last_token_{args['last_token_only']}_layer_{layers}.csv"
        )

        os.makedirs(output_dir, exist_ok=True)

        pd_result = pd.DataFrame(generation_result)
        pd_result.to_csv(file_path, index=False)


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("activations_count")

    args = parse_args()
    main(args)
