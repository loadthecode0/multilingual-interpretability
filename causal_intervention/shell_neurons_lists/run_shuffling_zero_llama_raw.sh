#!/usr/bin/env bash

# Sample shell script for shuffling experiment zero ablation for llama-raw
# Usage: bash multilingual_interpretability/causal_intervention/shell_neurons_lists/run_shuffling_zero_llama_raw.sh [category] [layers...]
# Default category: overlap
# Default layers: all (if none specified)

CATEGORY=${1:-overlap}
shift || true

# Remaining args are layers (e.g. 8 10 12). If empty, the python script defaults to all layers.
LAYERS="$@"

MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
MODEL_NAME="Llama-3.2-1B"
LANG="ru"
NEURON_LIST_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/neuron_lists"

# Set these to pass through to random sampling mode if needed
# NUM_RANDOM="--num-random-neurons 3" # e.g. "--num-random-neurons 50"
# SEED="--seed 0"       # e.g. "--seed 42"

NUM_RANDOM=""
SEED=""
echo "Running simultaneous causal zero ablation for shuffling category: $CATEGORY from $NEURON_LIST_ROOT"

if [ -z "$LAYERS" ]; then
    echo "Targeting ALL layers simultaneously"
    LAYER_ARG=""
else
    echo "Targeting layers simultaneously: $LAYERS"
    LAYER_ARG="--layer"
    for L in $LAYERS; do
        if [[ $L =~ ^[0-9]+$ ]]; then
            LAYER_ARG="$LAYER_ARG layers.$L"
        else
            LAYER_ARG="$LAYER_ARG $L"
        fi
    done
fi

# Run from the project root (e.g. ~/mi)
CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention_neuron_lists.py \
    --model_path "$MODEL" \
    --model_name "$MODEL_NAME" \
    $LAYER_ARG \
    --device "cuda" \
    --raw-model \
    --neuron-list-root "$NEURON_LIST_ROOT" \
    --experiment-type "shuffling" \
    --category "$CATEGORY" \
    --ablation "zero" \
    --lang "$LANG" \
    --max-examples 100 \
    --output-root "causal_results_neuron_lists_100ex" \
    $NUM_RANDOM \
    $SEED
