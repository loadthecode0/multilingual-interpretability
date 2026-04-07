#!/usr/bin/env bash

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
SAE="gemma-scope-2b-pt-mlp-canonical"
DOM="${MI_PROJECT_ROOT:-$(pwd)}/probing/dominance/gemma_results_new_cv"

LANG=hi
FEATURES=(syntax phonology inventory control)

for L in {10..25}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Gemma-2-2b" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation mean \
      --mean-source same \
      --lang "$LANG" \
      --top-k 50 \
      --gemma-scope-width 65k \
      --gemma-scope-l0 52
  done
done












