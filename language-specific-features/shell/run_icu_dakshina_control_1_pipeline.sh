#!/usr/bin/env bash
set -euo pipefail

# ICU + Dakshina Control Pipeline
# Splits native data into 500-example blocks and runs the full pipeline per split.
# Outputs are written to <OUT_ROOT>/<split_name>
#
# Usage:
#   bash ./shell/run_icu_dakshina_control_pipeline.sh

SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
HF_MODEL_ID="meta-llama/Llama-3.2-1B"
SAE_MODEL="EleutherAI/sae-Llama-3.2-1B-131k"
LOCAL_SAE_DIR="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
DATASET="icu_dakshina"
OUT_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output_romanization_expt_control_1"
SPLIT_NAME="dev"

LAYER_RANGE="model.layers.{0..15}.mlp"
LANGS_LIST=(hi mr bn ur ru bg ja zh ko en es)
LANGS_BRACED="{hi,mr,bn,ur,ru,bg,ja,zh,ko,en,es}"

# Define native splits: two 500-example blocks
SPLITS=(
  "split_0_500:0:500"
  "split_500_1000:500:1000"
)

mkdir -p "$OUT_ROOT"

for split_def in "${SPLITS[@]}"; do
  IFS=":" read -r SPLIT_ID START_IDX END_IDX <<<"$split_def"
  OUTPUT_DIR="$OUT_ROOT/$SPLIT_ID"
  mkdir -p "$OUTPUT_DIR"

  echo "=========================================="
  echo "ICU + Dakshina Control Pipeline"
  echo "Split: $SPLIT_ID ($START_IDX:$END_IDX)"
  echo "Output dir: $OUTPUT_DIR"
  echo "=========================================="

  # 1) activations_to_sae_features
  echo "[1/5] Collecting SAE features..."
  python3 "$SCRIPT_DIR/scripts/activations_to_sae_features.py" "$MODEL_PATH" "$DATASET" \
    --split "$SPLIT_NAME" \
    --lang "${LANGS_LIST[@]}" \
    --layer "$LAYER_RANGE" \
    --start "$START_IDX" \
    --end "$END_IDX" \
    --sae-model "$SAE_MODEL" \
    --batch 500 \
    --local-sae-dir "$LOCAL_SAE_DIR" \
    --out-dir "$OUTPUT_DIR"

  # 2) sae_statistics
  echo "[2/5] Computing SAE statistics..."
  python3 "$SCRIPT_DIR/scripts/sae_statistics.py" "$HF_MODEL_ID" "$DATASET" \
    --lang "${LANGS_LIST[@]}" \
    --layer "$LAYER_RANGE" \
    --sae-model "$SAE_MODEL" \
    --in-dir "$OUTPUT_DIR" \
    --out-dir "$OUTPUT_DIR"

  # 3) sae_features_count
  echo "[3/5] Counting SAE feature activations..."
  python3 "$SCRIPT_DIR/scripts/sae_features_count.py" \
    --output-type "EncoderOutput" \
    --hidden-dim 131072 \
    --dataset-configs "$DATASET:$LANGS_BRACED" \
    --layer "$LAYER_RANGE" \
    --in-path "$OUTPUT_DIR/sae_features/Llama-3.2-1B/$SAE_MODEL" \
    --out-path "$OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL"

  # 4) Identify language-specific features
  echo "[4/5] Running identify (language-specific)..."
  python3 "$SCRIPT_DIR/scripts/identify.py" \
    --model "$HF_MODEL_ID" \
    --sae-model "$SAE_MODEL" \
    --layer "$LAYER_RANGE" \
    --dataset-configs "$DATASET:$LANGS_BRACED" \
    --in-path "$OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --out-path "$OUTPUT_DIR/sae_features_specific/Llama-3.2-1B/$SAE_MODEL" \
    --out-filename "lape_all.pt" \
    --topk-threshold-ratio 0.5 \
    --example-rate 0.98 \
    --algorithm "sae_lape" \
    --lang-specific

  # 5) Identify shared features
  echo "[5/5] Running identify (shared sets)..."
  for count in {2..15}; do
    python3 "$SCRIPT_DIR/scripts/identify.py" \
      --model "$HF_MODEL_ID" \
      --sae-model "$SAE_MODEL" \
      --layer "$LAYER_RANGE" \
      --dataset-configs "$DATASET:$LANGS_BRACED" \
      --in-path "$OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
      --out-path "$OUTPUT_DIR/sae_features_shared/Llama-3.2-1B/$SAE_MODEL" \
      --out-filename "lape_shared_${count}.pt" \
      --topk-threshold-ratio 0.5 \
      --example-rate 0.98 \
      --algorithm "sae_lape" \
      --lang-shared \
      --shared-count "$count"
  done

  echo "✓ Completed split: $SPLIT_ID"
done

echo "All control splits completed. Outputs in $OUT_ROOT"


