#!/usr/bin/env bash

# Sample shell script for English and Hindi cross-mean ablation for llama-raw
# This script ablates neurons in specified layers of Llama-1B using the mean difference between English and Hindi.
# Usage: bash multilingual_interpretability/causal_intervention/shell_neurons_lists/run_cross_mean_llama_raw.sh [category] [layers...]
# Default category: overlap
# Default layers: all (if none specified)

CATEGORY=${1:-overlap}
shift || true

# Remaining args are layers (e.g. 8 10 12). If empty, the python script defaults to all layers.
LAYERS="$@"

MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
MODEL_NAME="Llama-3.2-1B"
LANG="en"
MEAN_LANG="hi"
NEURON_LIST_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/neuron_lists"

echo "Running simultaneous causal ablation for category: $CATEGORY from $NEURON_LIST_ROOT"

if [ -z "$LAYERS" ]; then
    echo "Targeting ALL layers simultaneously"
    LAYER_ARG=""
else
    echo "Targeting layers simultaneously: $LAYERS"
    # Prefix each layer with 'layers.' if it's just a number
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
# Note: Using the path path preferred by the user in recent edits
CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention_neuron_lists.py \
    --model_path "$MODEL" \
    --model_name "$MODEL_NAME" \
    $LAYER_ARG \
    --device "cuda" \
    --raw-model \
    --neuron-list-root "$NEURON_LIST_ROOT" \
    --experiment-type "romanization" \
    --romanization-mode "roman_dia" \
    --category "$CATEGORY" \
    --ablation "mean" \
    --mean-source "other" \
    --mean-lang "$MEAN_LANG" \
    --lang "$LANG" \
    --max-examples 100 \
    --output-root "causal_results_neuron_lists_100ex" \
    # --num-random-neurons 6 \

