#!/usr/bin/env bash

MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
SAE="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
DOM="${MI_PROJECT_ROOT:-$(pwd)}/probing/dominance/results_raw_cv"
LANG=en
# FEATURES=(syntax phonology inventory control)
FEATURES=(top_r2_all)

for L in 0 5 10 15; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation mean \
      --mean-source other \
      --mean-lang hi \
      --lang "$LANG" \
      --raw-model
  done
done
