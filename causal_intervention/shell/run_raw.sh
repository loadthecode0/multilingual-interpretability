#!/usr/bin/env bash
MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
SAE="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
DOM="${MI_PROJECT_ROOT:-$(pwd)}/probing/dominance/results_raw_cv"

LANG=hi
FEATURES=(syntax phonology inventory control)

for L in {0..15}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang "$LANG" \
      --raw-model
  done
done

for L in {0..15}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang "$LANG" \
      --raw-model \
      --neuron-mode random \
      --seed 42
  done
done

for L in {0..15}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang en \
      --raw-model
  done
done

for L in {0..15}; do
  for FEAT in "${FEATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python causal_intervention/run_causal_intervention.py \
      --model_path "$MODEL" \
      --model_name "Llama-3.2-1B" \
      --sae-model "$SAE" \
      --layer "layers.${L}.mlp" \
      --dominance-file "${DOM}/layer_${L}/dominance_${FEAT}.csv" \
      --feature-set "$FEAT" \
      --ablation zero \
      --lang en \
      --neuron-mode random \
      --seed 42 \
      --raw-model
  done
done
