import gc
import logging
import os
from functools import reduce
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import login
from const import model_name_to_display
from loader import load_env
from tqdm.auto import tqdm
from transformers import set_seed


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

    @staticmethod
    def get_logger(name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = TqdmLoggingHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger


def set_deterministic(seed: int = 42):
    # Set seed for reproducibility
    set_seed(seed=42, deterministic=True)

    # NNsightError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
    # To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
    # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def login_to_huggingface(env: Literal["local", "colab", "kaggle"] = "local"):
    HF_TOKEN = load_env("HF_TOKEN", env)

    login(token=HF_TOKEN)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_nested_attr(obj, attr_path):
    return reduce(getattr, attr_path.split("."), obj)


def get_project_dir():
    return Path(__file__).parent.parent


def get_model_dirname(model: str | None) -> str:
    if not model:
        return ""
    return model_name_to_display.get(model, model.split("/")[-1])


def clear_gpu_memory():
    # Invoke garbage collector
    gc.collect()

    # Clear GPU cache
    torch.cuda.empty_cache()
