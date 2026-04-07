import os
from pathlib import Path

def get_env_or_default(var_name, default):
    return os.environ.get(var_name, default)

# Base directory for models and SAEs
MODELS_DIR = get_env_or_default("MI_MODELS_DIR", "/home/models")

# Project root directory - default to the parent of the utils directory
SUBMISSION_ROOT = str(Path(__file__).resolve().parent.parent)
PROJECT_ROOT = get_env_or_default("MI_PROJECT_ROOT", SUBMISSION_ROOT)

# Legacy absolute path mapping for anonymization
# These can be overridden by environment variables
ANONYMOUS_HOME = get_env_or_default("MI_ANONYMOUS_HOME", "/home/user")

# Specific model paths
LLAMA_1B_PATH = os.path.join(MODELS_DIR, "meta-llama_Llama-3.2-1B")
GEMMA_2B_PATH = os.path.join(MODELS_DIR, "gemma-2-2b")
LLAMA_1B_SAE_DIR = os.path.join(MODELS_DIR, "sae-Llama-3.2-1B-131k")

# Data and results directories (relative to PROJECT_ROOT where possible)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DAKSHINA_DIR = os.path.join(DATA_DIR, "multilingual_datasets/dakshina")
FLORES_ROMANIZED_DIR = os.path.join(DATA_DIR, "multilingual_datasets/flores_plus_romanized")
NEURON_LISTS_DIR = os.path.join(PROJECT_ROOT, "neuron_lists")
DOMINANCE_DIR = os.path.join(PROJECT_ROOT, "probing/dominance")

# Legacy/Alternative names for backward compatibility with some scripts
DAKSHINA_DATASET_V1 = DAKSHINA_DIR
FLORES_PLUS_ROMANIZED_RESULTS = FLORES_ROMANIZED_DIR

# Language-specific features outputs
LAPE_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "language-specific-features")
LAPE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "language-specific-features/outputs")

def anonymize_path(path):
    """Replaces common hardcoded prefixes with anonymized versions or relative paths."""
    if not path:
        return path
    
    # Replace the specific user's home directory if it appears
    path = path.replace("/home/user/mi", PROJECT_ROOT)
    path = path.replace("/home/user/multilingual_interpretability", PROJECT_ROOT)
    path = path.replace("/home/user", ANONYMOUS_HOME)
    
    return path
