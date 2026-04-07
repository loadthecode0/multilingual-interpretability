#!/usr/bin/env bash

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
SAE="gemma-scope-2b-pt-mlp-canonical"
DOM="${MI_PROJECT_ROOT:-$(pwd)}/probing/dominance/gemma_results_new_cv"

LANGS=(hi en)
FEATURES=(syntax phonology inventory control)

# Deterministic zero ablations for each language
for LANG in "${LANGS[@]}"; do
  for L in {0..25}; do
    for FEAT in "${FEATURES[@]}"; do
      CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
        --model_path "$MODEL" \
        --model_name "Gemma-2-2b" \
        --sae-model "$SAE" \
        --layer "layers.${L}.mlp" \
        --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
        --feature-set "$FEAT" \
        --ablation zero \
        --lang "$LANG" \
        --gemma-scope-width 65k \
        --gemma-scope-l0 52
    done
  done
done

# Random-matched ablations for English as a control
for L in {0..25}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Gemma-2-2b" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang en \
      --neuron-mode random \
      --seed 42 \
      --gemma-scope-width 65k \
      --gemma-scope-l0 52
  done
done












