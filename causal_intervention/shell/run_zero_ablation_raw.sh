#!/usr/bin/env bash
MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
SAE="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
DOM="dominance/Llama-3.2-1B/flores_plus"

LANG=hi
FEATURES=(syntax_wals phonology_wals fam geo)

for L in {0..15}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang "$LANG" \
      --raw-model
  done
done
