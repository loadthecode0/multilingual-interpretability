#!/usr/bin/env bash
set -euo pipefail

# ICU + Dakshina Pipeline
# Languages:
#   Indic (Dakshina): hi, mr, bn, ur
#   ICU-FLORES JSONL: ru, bg, ja, zh, ko, en, es
#
# Usage:
#   bash ./shell/run_icu_dakshina_pipeline.sh [--romanized]
# Example:
#   bash ./shell/run_icu_dakshina_pipeline.sh
#   bash ./shell/run_icu_dakshina_pipeline.sh --romanized

SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
OUTPUT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output_romanization_expt"
HF_MODEL_ID="meta-llama/Llama-3.2-1B"
SAE_MODEL="EleutherAI/sae-Llama-3.2-1B-131k"
MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
LOCAL_SAE_DIR="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
DATASET="icu_dakshina"
ROMANIZED_FLAG="${1:-}"
LAYER_RANGE="model.layers.{0..15}.mlp"
LANGS="{hi,mr,bn,ur,ru,bg,ja,zh,ko,en,es}"
LANGS_ARR=(hi mr bn ur ru bg ja zh ko en es)
SPLIT="dev"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ICU + Dakshina Pipeline"
echo "Dataset: $DATASET"
echo "Languages: $LANGS"
echo "Romanized: ${ROMANIZED_FLAG:-false}"
echo "=========================================="

# 1) activations_to_sae_features (native or romanized depending on flag)
echo "[1/4] Collecting SAE features..."
python3 "$SCRIPT_DIR/scripts/activations_to_sae_features.py" "$MODEL_PATH" "$DATASET" \
  --split "$SPLIT" \
  --lang "${LANGS_ARR[@]}" \
  --layer "$LAYER_RANGE" \
  --start 0 --end 1000 \
  --sae-model "$SAE_MODEL" \
  --batch 500 \
  --local-sae-dir "$LOCAL_SAE_DIR" \
  --out-dir "$OUTPUT_DIR" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# 2) sae_statistics (native or romanized depending on flag)
echo "[2/4] Computing SAE statistics..."
python3 "$SCRIPT_DIR/scripts/sae_statistics.py" "$HF_MODEL_ID" "$DATASET" \
  --lang "${LANGS_ARR[@]}" \
  --layer "$LAYER_RANGE" \
  --sae-model "$SAE_MODEL" \
  --in-dir "$OUTPUT_DIR" \
  --out-dir "$OUTPUT_DIR" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# Step: Count SAE feature activations
echo "[3/4] Counting SAE feature activations..."
python3 "$SCRIPT_DIR/scripts/sae_features_count.py" \
  --output-type "EncoderOutput" \
  --hidden-dim 131072 \
  --dataset-configs "$DATASET:$LANGS" \
  --layer "$LAYER_RANGE" \
  --in-path "./output_romanization_expt/sae_features/Llama-3.2-1B/$SAE_MODEL" \
  --out-path "./output_romanization_expt/sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# Step: Identify language-specific (all) features
echo "[4/5] Running identify (LAPE) for language-specific features..."
python3 "$SCRIPT_DIR/scripts/identify.py" \
  --model "$HF_MODEL_ID" \
  --sae-model "$SAE_MODEL" \
  --layer "$LAYER_RANGE" \
  --dataset-configs "$DATASET:$LANGS" \
  --in-path './output_romanization_expt/sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
  --out-path './output_romanization_expt/sae_features_specific/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
  --out-filename 'lape_all.pt' \
  --topk-threshold-ratio 0.5 \
  --example-rate 0.98 \
  --algorithm 'sae_lape' \
  --lang-specific \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# Step: Identify shared features (shared counts sweep)
echo "[5/5] Running identify (LAPE) on shared counts..."
for count in {2..15}; do
  python3 "$SCRIPT_DIR/scripts/identify.py" \
    --model "$HF_MODEL_ID" \
    --sae-model "$SAE_MODEL" \
    --layer "$LAYER_RANGE" \
    --dataset-configs "$DATASET:$LANGS" \
    --in-path "./output_romanization_expt/sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --out-path "./output_romanization_expt/sae_features_shared/Llama-3.2-1B/$SAE_MODEL" \
    --out-filename "lape_shared_${count}.pt" \
    --topk-threshold-ratio 0.5 \
    --example-rate 0.98 \
    --lang-shared \
    --shared-count "$count" \
    --algorithm "sae_lape" \
    $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)
done

echo "✓ ICU + Dakshina pipeline completed."
echo "Outputs in: $OUTPUT_DIR"


