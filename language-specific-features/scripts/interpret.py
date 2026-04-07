import asyncio
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import orjson
import torch
from const import hookpoint_to_layer
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, RunConfig
from delphi.explainers import ContrastiveExplainer, DefaultExplainer
from delphi.latents import LatentCache, LatentDataset
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.sparse_coders import load_sparse_coders
from delphi.sparse_coders.load_sparsify import resolve_path, sae_dense_latents
from delphi.utils import assert_type
from loader import load_env, load_from_dataset_configs
from simple_parsing import ArgumentParser, list_field
from sparsify import Sae
from sparsify.data import chunk_and_tokenize
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from utils import TqdmLoggingHandler


@dataclass
class CustomCacheConfig(CacheConfig):
    dataset_configs: list[str] = list_field()
    """
    List of dataset configurations. Support bracex expansion.
    Each configuration is a string with the following format:
    "dataset_name:config_name:split_name"
    """
    dataset_start: int = 0
    """
    The starting index of the dataset configuration to use."""

    dataset_end: int = 1000
    """
    The ending index of the dataset configuration to use.
    """


@dataclass
class CustomRunConfig(RunConfig):
    cache_cfg: CustomCacheConfig
    """Cache configuration."""
    lape_result_path: Path = None
    """Path to the LAPE result file."""
    output_dir: Path = Path.cwd()
    """Path to the output directory."""
    env: Literal["local", "colab", "kaggle"] = "local"
    """Environment to use for loading the OpenRouter API key."""
    populate_cache: bool = False
    """Whether to populate the cache with activations."""
    scoring: bool = False
    """Whether to score the activations."""
    log_results: bool = False
    """Whether to log the results."""


def load_sae_custom(run_cfg: CustomRunConfig, model):
    sparse_model_dict = Sae.load_many(
        run_cfg.sparse_model,
        layers=run_cfg.hookpoints,
        device=model.device,
    )

    hookpoint_to_sparse_encode = {}
    transcode = False

    for hookpoint, sparse_model in sparse_model_dict.items():
        print(f"Resolving path for hookpoint: {hookpoint}")
        path_segments = resolve_path(model, hookpoint.split("."))

        if path_segments is None:
            raise ValueError(f"Could not find valid path for hookpoint: {hookpoint}")

        hookpoint_to_sparse_encode[".".join(path_segments)] = partial(
            sae_dense_latents, sae=sparse_model
        )
        # We only need to check if one of the sparse models is a transcoder
        if hasattr(sparse_model.cfg, "transcode"):
            if sparse_model.cfg.transcode:
                transcode = True
        if hasattr(sparse_model.cfg, "skip_connection"):
            if sparse_model.cfg.skip_connection:
                transcode = True

    return hookpoint_to_sparse_encode, transcode


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    hookpoint_to_sparse_encode, transcode = load_sae_custom(run_cfg, model)

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if constructor_cfg.neighbours_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:

        if constructor_cfg.neighbours_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "decoder_similarity":

            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: CustomRunConfig,
    latents_path: Path,
    prompts_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_dict: dict[str, Tensor] = None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `fuzz_scores_path` directory.
    """
    prompts_path.mkdir(parents=True, exist_ok=True)
    explanations_path.mkdir(parents=True, exist_ok=True)

    fuzz_scores_path = scores_path / "fuzz"
    detection_scores_path = scores_path / "detection"
    fuzz_scores_path.mkdir(parents=True, exist_ok=True)
    detection_scores_path.mkdir(parents=True, exist_ok=True)

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    if run_cfg.explainer_provider == "offline":
        client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
        )
    elif run_cfg.explainer_provider == "openrouter":

        client = OpenRouter(
            run_cfg.explainer_model,
            api_key=load_env("OPENROUTER_API_KEY", run_cfg.env),
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    def explainer_postprocess(result):
        with open(explanations_path / f"{result.record.latent}.txt", "wb") as f:
            f.write(orjson.dumps(result.explanation))

        return result

    if run_cfg.constructor_cfg.non_activating_source == "FAISS":
        explainer = ContrastiveExplainer(
            client,
            threshold=0.3,
            verbose=run_cfg.verbose,
        )
    else:
        explainer = DefaultExplainer(
            client,
            threshold=0.3,
            verbose=run_cfg.verbose,
            prompts_path=prompts_path,
        )

    explainer_pipe = Pipe(process_wrapper(explainer, postprocess=explainer_postprocess))

    # Builds the record from result returned by the pipeline
    def scorer_preprocess(result):
        if isinstance(result, list):
            result = result[0]

        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    # Saves the score to a file
    def scorer_postprocess(result, score_dir):
        safe_latent_name = str(result.record.latent).replace("/", "--")

        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = Pipe(
        process_wrapper(
            DetectionScorer(
                client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=detection_scores_path),
        ),
        process_wrapper(
            FuzzingScorer(
                client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=fuzz_scores_path),
        ),
    )

    pipeline = Pipeline(
        dataset,
        explainer_pipe,
        scorer_pipe,
    )

    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    await pipeline.run(run_cfg.pipeline_num_proc)


def load_tokenized_data_custom(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_configs: str,
    dataset_start: int,
    dataset_end: int,
    column_name: str = "text",
):
    """
    Load a huggingface dataset from dataset config, tokenize it, and shuffle.
    Using this function ensures we are using the same tokens everywhere.
    """

    data = load_from_dataset_configs(
        dataset_configs, dataset_start, dataset_end, column_name
    )

    logger.info(f"Loaded {len(data)} samples from {dataset_configs}")
    logger.info(f"First 10 data points: {data['text'][:10]}")
    logger.info(f"Last 10 data points: {data['text'][-10:]}")

    tokens_ds = chunk_and_tokenize(
        data,  # type: ignore
        tokenizer,
        max_seq_len=ctx_len,
        text_key=column_name,
    )

    tokens = tokens_ds["input_ids"]

    logger.info(
        f"Tokenized data shape: {tokens.shape}, num tokens: {tokens.size(0) * tokens.size(1)}"
    )

    return tokens


def populate_cache(
    run_cfg: CustomRunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg

    tokens = load_tokenized_data_custom(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_configs,
        cache_cfg.dataset_start,
        cache_cfg.dataset_end,
        cache_cfg.dataset_column,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode

    in_results_path = [x.name for x in results_path.glob("*")]

    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]

    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")

    return non_redundant_hookpoints


async def run(
    run_cfg: CustomRunConfig,
):
    # Create a base path for the results
    base_path = run_cfg.output_dir

    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)

    # Save the run configuration to a JSON file
    run_cfg.save_json(base_path / "run_config.json", indent=4)

    # Create directories for the results
    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"
    prompts_path = base_path / "prompts"

    # Load LAPE
    lape = torch.load(
        run_cfg.lape_result_path,
        weights_only=False,
    )

    hookpoint_to_layer_indicies = {}

    for hookpoint in run_cfg.hookpoints:
        layer_index = hookpoint_to_layer[hookpoint]

        layer_indicies = []

        for lang_indices in lape["final_indice"]:
            layer_indicies.extend(lang_indices[layer_index].tolist())

        layer_indicies.sort()

        hookpoint_to_layer_indicies[hookpoint] = layer_indicies

    # latents to be explained
    hookpoint_to_selected_latents = {
        hookpoint: torch.tensor(hookpoint_to_layer_indicies[hookpoint])
        for hookpoint in run_cfg.hookpoints
    }

    logger.info(f"Latents to be explained: {hookpoint_to_selected_latents}")

    # # Load model
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    if run_cfg.populate_cache:
        # Don't overwrite already existing cache files
        hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(
            run_cfg
        )
        nrh = assert_type(
            dict,
            non_redundant_hookpoints(
                hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
            ),
        )

        # Populate the cache with activations
        if nrh:
            populate_cache(
                run_cfg,
                model,
                nrh,
                latents_path,
                tokenizer,
                transcode,
            )

        del model, hookpoint_to_sparse_encode  # Free up memory

        # Neighbours creation
        if run_cfg.constructor_cfg.non_activating_source == "neighbours":
            nrh = assert_type(
                list,
                non_redundant_hookpoints(
                    hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite
                ),
            )
            if nrh:
                create_neighbours(
                    run_cfg,
                    latents_path,
                    neighbours_path,
                    nrh,
                )
        else:
            print("Skipping neighbour creation")
    else:
        hookpoints = run_cfg.hookpoints

    if run_cfg.scoring:
        # Scores creation
        nrh = assert_type(
            list,
            non_redundant_hookpoints(
                hookpoints, scores_path, "scores" in run_cfg.overwrite
            ),
        )
        if nrh:
            await process_cache(
                run_cfg,
                latents_path,
                prompts_path,
                explanations_path,
                scores_path,
                nrh,
                tokenizer,
                hookpoint_to_selected_latents,
            )

    if run_cfg.verbose or run_cfg.log_results:
        log_results(scores_path, visualize_path, run_cfg.hookpoints)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CustomRunConfig, dest="run_cfg")
    args = parser.parse_args()

    logger = TqdmLoggingHandler.get_logger("interpret")

    asyncio.run(run(args.run_cfg))
