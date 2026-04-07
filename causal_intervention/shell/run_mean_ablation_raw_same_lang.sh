#!/usr/bin/env bash
for L in {0..15}; do
  CUDA_VISIBLE_DEVICES=2 python run_causal_intervention.py \
    --model_path "$MODEL" \
    --model_name "Llama-3.2-1B" \
    --sae-model "$SAE" \
    --layer "layers.${L}.mlp" \
    --dominance-file "${DOM}/layer_${L}.csv" \
    --feature-set syntax_wals \
    --ablation mean \
    --mean-source same \
    --lang hi \
    --raw-model
done
