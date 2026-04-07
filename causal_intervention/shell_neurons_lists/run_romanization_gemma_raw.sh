#!/usr/bin/env bash

# Romanization interventions for Gemma-2-2b raw
# Runs hi->en and en->hi, both only_native and overlap modes

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
MODEL_NAME="Gemma-2-2b"
NEURON_LIST_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/neuron_lists"

CATEGORIES=("only_native" "overlap")

# hi -> en (eval on en, mean from hi)
for CAT in "${CATEGORIES[@]}"; do
    echo "--------------------------------------------------------"
    echo "Running Romanization Gemma Raw: hi -> en, Category: $CAT"
    echo "--------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention_neuron_lists.py \
        --model_path "$MODEL" \
        --model_name "$MODEL_NAME" \
        --device "cuda" \
        --raw-model \
        --neuron-list-root "$NEURON_LIST_ROOT" \
        --experiment-type "romanization" \
        --romanization-mode "roman_dia" \
        --category "$CAT" \
        --ablation "mean" \
        --mean-source "other" \
        --mean-lang "hi" \
        --lang "en" \
        --max-examples 100 \
        --output-root "causal_results_neuron_lists_100ex"
done

# en -> hi (eval on hi, mean from en)
for CAT in "${CATEGORIES[@]}"; do
    echo "--------------------------------------------------------"
    echo "Running Romanization Gemma Raw: en -> hi, Category: $CAT"
    echo "--------------------------------------------------------"
    CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention_neuron_lists.py \
        --model_path "$MODEL" \
        --model_name "$MODEL_NAME" \
        --device "cuda" \
        --raw-model \
        --neuron-list-root "$NEURON_LIST_ROOT" \
        --experiment-type "romanization" \
        --romanization-mode "roman_dia" \
        --category "$CAT" \
        --ablation "mean" \
        --mean-source "other" \
        --mean-lang "en" \
        --lang "hi" \
        --max-examples 100 \
        --output-root "causal_results_neuron_lists_100ex"
done

