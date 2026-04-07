#!/usr/bin/env bash
set -euo pipefail

# Native vs Romanized LAPE pipeline (non-SAE) for icu_dakshina
#
# This script runs the neuron-level LAPE identification flow twice:
#   1) Native script inputs
#   2) Romanized inputs (ICU transliteration)
# Each run collects MLP activation statistics and then identifies both
# language-specific and shared neurons via the plain `lape` metric.
#
# Usage:
#   bash ./shell/run_icu_dakshina_lape_pipeline.sh

REPO_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
MODEL_ID="meta-llama/Llama-3.2-1B"
LOCAL_MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
DATASET="icu_dakshina"
LANG_SET="{hi,mr,bn,ur,ru,bg,ja,zh,ko,en,es}"
SPLIT="dev"
START_IDX=0
END_IDX=1000
HIDDEN_DIM=8192
ACT_LAYERS="model.layers.{0..15}.mlp.act_fn"
IDENTIFY_LAYERS="model.layers.{0..15}.mlp"
OUTPUT_ROOT="$REPO_ROOT/output_romanization_expt_raw_no_diacritics"
ACTS_SUBDIR="mlp_acts_count/Llama-3.2-1B"
SPEC_SUBDIR="mlp_acts_specific/Llama-3.2-1B"
SHARED_SUBDIR="mlp_acts_shared/Llama-3.2-1B"
SHARED_MIN=2
SHARED_MAX=15

mkdir -p \
  "$OUTPUT_ROOT/$ACTS_SUBDIR" \
  "$OUTPUT_ROOT/$SPEC_SUBDIR" \
  "$OUTPUT_ROOT/$SHARED_SUBDIR"

run_mode() {
  local mode_label="$1"      # "Native" or "Romanized"
  local romanized_flag="$2"  # "" or "--romanized"

  echo "=========================================="
  echo "ICU Dakshina LAPE Pipeline - ${mode_label}"
  echo "Dataset: $DATASET"
  echo "Languages: $LANG_SET"
  echo "Romanized flag: ${romanized_flag:-<none>}"
  echo "=========================================="

  echo "[1/3] Collecting activation counts ($mode_label)..."
  python3 "$REPO_ROOT/scripts/activations_count.py" "$MODEL_ID" \
    --hidden-dim "$HIDDEN_DIM" \
    --dataset-configs "$DATASET:$LANG_SET:$SPLIT:$START_IDX:$END_IDX" \
    --layer "$ACT_LAYERS" \
    --out-dir "$OUTPUT_ROOT" \
    --out-path "$ACTS_SUBDIR" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    $romanized_flag

  echo "[2/3] Identifying language-specific neurons with LAPE ($mode_label)..."
  python3 "$REPO_ROOT/scripts/identify.py" \
    --model "$MODEL_ID" \
    --layer "$IDENTIFY_LAYERS" \
    --dataset-configs "$DATASET:$LANG_SET" \
    --in-dir "$OUTPUT_ROOT" \
    --in-path "$ACTS_SUBDIR" \
    --out-dir "$OUTPUT_ROOT" \
    --out-path "$SPEC_SUBDIR" \
    --out-filename "lape_neuron.pt" \
    --algorithm "lape" \
    --lang-specific \
    $romanized_flag

  echo "[3/3] Identifying shared neurons across language counts ($mode_label)..."
  for shared_count in $(seq "$SHARED_MIN" "$SHARED_MAX"); do
    python3 "$REPO_ROOT/scripts/identify.py" \
      --model "$MODEL_ID" \
      --layer "$IDENTIFY_LAYERS" \
      --dataset-configs "$DATASET:$LANG_SET" \
      --in-dir "$OUTPUT_ROOT" \
      --in-path "$ACTS_SUBDIR" \
      --out-dir "$OUTPUT_ROOT" \
      --out-path "$SHARED_SUBDIR" \
      --out-filename "lape_shared_${shared_count}.pt" \
      --algorithm "lape" \
      --lang-shared \
      --shared-count "$shared_count" \
      $romanized_flag
  done

  echo "✓ Completed ${mode_label} LAPE run."
  echo
}

# run_mode "Native" ""
run_mode "Romanized" "--romanized"

echo "All LAPE runs finished."

