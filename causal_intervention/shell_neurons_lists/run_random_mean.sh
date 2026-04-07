#!/usr/bin/env bash

# Random control interventions for Gemma-2-2b raw
# Usage: bash run_random_gemma_raw.sh [experiment_type] [lang] [num_random_list...]
# Example: bash run_random_gemma_raw.sh shuffling en 2 3 4 5

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
MODEL_NAME="Gemma-2-2b"

EXP_TYPE=$1
LANG=$2
MEAN_LANG=$3
shift 3
NUMS=$@

if [ -z "$EXP_TYPE" ] || [ -z "$LANG" ] || [ -z "$MEAN_LANG" ] || [ -z "$NUMS" ]; then
    echo "Usage: $0 [shuffling|romanization] [lang] [mean_lang] [num1 num2 ...]"
    exit 1
fi

for NUM in $NUMS; do
    echo "------------------------------------------------------------------------"
    echo "Running Random Control Gemma Raw: Exp: $EXP_TYPE, Lang: $LANG, Mean Lang: $MEAN_LANG, Num: $NUM"
    echo "------------------------------------------------------------------------"
    
    EXTRA_ARGS=""
    if [ "$EXP_TYPE" == "romanization" ]; then
        EXTRA_ARGS="--romanization-mode roman_dia"
    fi

    CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention_neuron_lists.py \
        --model_path "$MODEL" \
        --model_name "$MODEL_NAME" \
        --device "cuda" \
        --raw-model \
        --experiment-type "$EXP_TYPE" \
        --category "random" \
        --ablation "mean" \
        --mean-source "other" \
        --mean-lang "$MEAN_LANG" \
        --lang "$LANG" \
        --max-examples 100 \
        --output-root "causal_results_neuron_lists_100ex" \
        --num-random-neurons "$NUM" \
        --seed 0 \
        $EXTRA_ARGS
done

