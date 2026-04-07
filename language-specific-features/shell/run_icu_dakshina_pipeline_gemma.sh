#!/usr/bin/env bash
set -euo pipefail

# ICU + Dakshina Pipeline (Gemma-2-2b + GemmaScope SAE)
# Languages:
#   Indic (Dakshina): hi, mr, bn, ur
#   ICU-FLORES JSONL: ru, bg, ja, zh, ko, en, es
#
# Usage:
#   bash ./shell/run_icu_dakshina_pipeline_gemma.sh [--romanized]

SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
OUTPUT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output_romanization_expt_gemma_no_diacritics"
MODEL_PATH="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
HF_MODEL_ID="google/gemma-2-2b"
# Match folder naming used by activations_to_sae_features.py
MODEL_DIR_NAME="$(basename "$MODEL_PATH")"
SAE_MODEL="gemma-scope-2b-pt-mlp-canonical"
DATASET="icu_dakshina"
ROMANIZED_FLAG="${1:-}"
LAYER_RANGE="model.layers.{0..25}.mlp"
LANGS="{hi,mr,bn,ur,ru,bg,ja,zh,ko,en,es}"
LANGS_ARR=(hi mr bn ur ru bg ja zh ko en es)
SPLIT="dev"

FEATURE_DIR="$OUTPUT_DIR/sae_features/$MODEL_DIR_NAME/$SAE_MODEL"
COUNT_DIR="$OUTPUT_DIR/sae_features_count/$MODEL_DIR_NAME/$SAE_MODEL"
SPECIFIC_DIR="$OUTPUT_DIR/sae_features_specific/$MODEL_DIR_NAME/$SAE_MODEL"
SHARED_DIR="$OUTPUT_DIR/sae_features_shared/$MODEL_DIR_NAME/$SAE_MODEL"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ICU + Dakshina Pipeline (Gemma)"
echo "Dataset: $DATASET"
echo "Languages: $LANGS"
echo "Romanized: ${ROMANIZED_FLAG:-false}"
echo "=========================================="

# 1) activations_to_sae_features (native or romanized depending on flag)
echo "[1/5] Collecting SAE features..."
python3 "$SCRIPT_DIR/scripts/activations_to_sae_features.py" "$MODEL_PATH" "$DATASET" \
  --split "$SPLIT" \
  --lang "${LANGS_ARR[@]}" \
  --layer "$LAYER_RANGE" \
  --start 0 --end 1000 \
  --sae-model "$SAE_MODEL" \
  --batch 500 \
  --out-dir "$OUTPUT_DIR" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# 2) sae_statistics (native or romanized depending on flag)
echo "[2/5] Computing SAE statistics..."
python3 "$SCRIPT_DIR/scripts/sae_statistics.py" "$MODEL_PATH" "$DATASET" \
  --lang "${LANGS_ARR[@]}" \
  --layer "$LAYER_RANGE" \
  --sae-model "$SAE_MODEL" \
  --in-dir "$OUTPUT_DIR" \
  --out-dir "$OUTPUT_DIR" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# 3) sae_features_count
echo "[3/5] Counting SAE feature activations..."
python3 "$SCRIPT_DIR/scripts/sae_features_count.py" \
  --output-type "EncoderOutput" \
  --hidden-dim 65536 \
  --dataset-configs "$DATASET:$LANGS" \
  --layer "$LAYER_RANGE" \
  --in-path "$FEATURE_DIR" \
  --out-path "$COUNT_DIR" \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# 4) Identify language-specific (all) features
echo "[4/5] Running identify (LAPE) for language-specific features..."
python3 "$SCRIPT_DIR/scripts/identify.py" \
  --model "$HF_MODEL_ID" \
  --sae-model "$SAE_MODEL" \
  --layer "$LAYER_RANGE" \
  --dataset-configs "$DATASET:$LANGS" \
  --in-path "$COUNT_DIR" \
  --out-path "$SPECIFIC_DIR" \
  --out-filename 'lape_all.pt' \
  --topk-threshold-ratio 0.5 \
  --example-rate 0.98 \
  --algorithm 'sae_lape' \
  --lang-specific \
  $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)

# 5) Identify shared features across shared counts
echo "[5/5] Running identify (LAPE) on shared counts..."
for count in {2..15}; do
  python3 "$SCRIPT_DIR/scripts/identify.py" \
    --model "$HF_MODEL_ID" \
    --sae-model "$SAE_MODEL" \
    --layer "$LAYER_RANGE" \
    --dataset-configs "$DATASET:$LANGS" \
    --in-path "$COUNT_DIR" \
    --out-path "$SHARED_DIR" \
    --out-filename "lape_shared_${count}.pt" \
    --topk-threshold-ratio 0.5 \
    --example-rate 0.98 \
    --algorithm "sae_lape" \
    --lang-shared \
    --shared-count "$count" \
    $([ "${ROMANIZED_FLAG:-}" = "--romanized" ] && echo "--romanized" || true)
done

echo "✓ ICU + Dakshina pipeline (Gemma) completed."
echo "Outputs in: $OUTPUT_DIR"

