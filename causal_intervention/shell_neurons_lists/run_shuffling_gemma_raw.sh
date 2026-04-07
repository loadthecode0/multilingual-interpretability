#!/usr/bin/env bash

# Shuffling interventions for Gemma-2-2b raw
# Runs en, hi, zh, fr for both only_normal and overlap modes

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
MODEL_NAME="Gemma-2-2b"
NEURON_LIST_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/neuron_lists"

LANGS=("en" "hi" "zh" "fr")
CATEGORIES=("only_normal" "overlap")

for LANG in "${LANGS[@]}"; do
    for CAT in "${CATEGORIES[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running Shuffling Gemma Raw: Lang: $LANG, Category: $CAT"
        echo "--------------------------------------------------------"
        CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention_neuron_lists.py \
            --model_path "$MODEL" \
            --model_name "$MODEL_NAME" \
            --device "cuda" \
            --raw-model \
            --neuron-list-root "$NEURON_LIST_ROOT" \
            --experiment-type "shuffling" \
            --category "$CAT" \
            --ablation "zero" \
            --lang "$LANG" \
            --max-examples 100 \
            --output-root "causal_results_neuron_lists_100ex"
    done
done

