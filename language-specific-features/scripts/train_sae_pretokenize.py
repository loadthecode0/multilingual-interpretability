import math
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
from huggingface_hub import create_repo
from simple_parsing import field, parse
from sparsify.data import chunk_and_tokenize
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class RunConfig:
    model: str = field(
        default="meta-llama/Llama-3.2-1B",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="EleutherAI/rpj-v2-sample",
        positional=True,
    )
    """Path to the dataset to use for training."""

    batch_size: int = 100
    """Dataset batch size measured in sequences to be pretokenized."""

    subset: str | None = None
    """Subset of the dataset to use for training, if applicable."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""

    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    text_column: str = "text"
    """Column name to use for text data."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    cache_dir: str | None = None
    """Cache directory to use for loading datasets."""

    output_dir: Path = Path("pretokenized")
    """Output directory to save pretokenized datasets."""

    hf_username: str | None = None
    """Huggingface username to use for publishing pretokenized datasets."""

    hf_dataset_name: str | None = None
    """Huggingface dataset name to use for publishing pretokenized datasets."""


def pretokenize(args: RunConfig):
    print(
        f"Loading dataset '{args.dataset}' (split '{args.split}') from cache {args.cache_dir or 'default'}..."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)

    dataset_builder = load_dataset_builder(
        args.dataset,
        args.subset,
        cache_dir=args.cache_dir,
    )

    total_examples = dataset_builder.info.splits[args.split].num_examples

    print(f"Total examples in dataset: {total_examples}")

    num_partitions = math.ceil(total_examples / args.batch_size)

    print(f"Number of partitions: {num_partitions}")

    num_padding = len(str(num_partitions)) + 1

    for partition in tqdm(range(num_partitions), desc="Processing partitions"):
        start_idx = partition * args.batch_size
        end_idx = min(start_idx + args.batch_size, total_examples)

        dataset = load_dataset(
            args.dataset,
            args.subset,
            split=f"{args.split}[{start_idx}:{end_idx}]",
            cache_dir=args.cache_dir,
        )

        dataset = chunk_and_tokenize(
            dataset,
            tokenizer,
            max_seq_len=args.ctx_len,
            num_proc=args.data_preprocessing_num_proc,
            text_key=args.text_column,
        )

        # Save the pretokenized dataset to disk
        folder_name = f"pretokenized_{args.split}_{partition:0{num_padding}}"
        dataset.save_to_disk(args.output_dir / folder_name)


def merge_pretokenized(args: RunConfig):
    pretokenized_folder_names = sorted(args.output_dir.iterdir())

    print(f"Merging {len(pretokenized_folder_names)} pretokenized datasets...")
    print(pretokenized_folder_names)

    merged_dataset = None

    for folder_name in tqdm(
        pretokenized_folder_names, desc="Merging pretokenized datasets"
    ):
        if not folder_name.is_dir():
            continue

        dataset = load_from_disk(folder_name)

        if merged_dataset is None:
            merged_dataset = dataset
        else:
            merged_dataset = concatenate_datasets(
                [
                    merged_dataset,
                    dataset,
                ]
            )

    final_merged_dataset = DatasetDict({"train": merged_dataset})

    # Save the merged dataset to disk
    folder_name = f"merged_pretokenized_{args.split}"
    final_merged_dataset.save_to_disk(args.output_dir / folder_name)

    return final_merged_dataset


def publish_pretokenized(args: RunConfig, dataset: DatasetDict):
    repo_id = f"{args.hf_username}/{args.hf_dataset_name}"

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=args.hf_token,
    )

    dataset.push_to_hub(
        repo_id=repo_id,
        token=args.hf_token,
    )


def main(args: RunConfig):
    pretokenize(args)
    merged_dataset = merge_pretokenized(args)
    publish_pretokenized(args, merged_dataset)


if __name__ == "__main__":
    args = parse(RunConfig)
    main(args)
