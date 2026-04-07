#!/usr/bin/env bash
# Combined LAPE pipeline (meta-llama/Llama-3.2-1B) that runs once with the
# original datasets and once with word-shuffled prompts. Each run collects
# neuron activation counts and then identifies language-specific and shared
# neurons via the plain LAPE algorithm.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"

MODEL_ID="meta-llama/Llama-3.2-1B"
LOCAL_MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
OUT_DIR="$ROOT/output_lape_shuffling"

HIDDEN_DIM=2048
ACT_LAYERS="model.layers.{0..15}.mlp.act_fn"
IDENTIFY_LAYERS="model.layers.{0..15}.mlp"

ACTS_SUBDIR="mlp_acts_count/Llama-3.2-1B"
SPEC_SUBDIR="mlp_acts_specific/Llama-3.2-1B"
SHARED_SUBDIR="mlp_acts_shared/Llama-3.2-1B"

# Datasets and languages (match existing SAE-based shuffled pipeline)
XNLI_DATASET="facebook/xnli"
XNLI_LANGS="{en,de,fr,hi,es,th,bg,ru,tr,vi}"

PAWSX_DATASET="google-research-datasets/paws-x"
PAWSX_LANGS="{en,de,fr,es,ja,ko,zh}"

FLORES_DATASET="openlanguagedata/flores_plus"
FLORES_LANGS="{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}"

COUNT_CONFIGS=(
  "$XNLI_DATASET:$XNLI_LANGS:train:0:1000"
  "$PAWSX_DATASET:$PAWSX_LANGS:train:0:1000"
  "$FLORES_DATASET:$FLORES_LANGS:dev:0:997"
)

IDENTIFY_CONFIGS=(
  "$XNLI_DATASET:$XNLI_LANGS"
  "$PAWSX_DATASET:$PAWSX_LANGS"
  "$FLORES_DATASET:$FLORES_LANGS"
)

mkdir -p \
  "$OUT_DIR/$ACTS_SUBDIR" \
  "$OUT_DIR/$SPEC_SUBDIR" \
  "$OUT_DIR/$SHARED_SUBDIR"

run_mode() {
  local mode_label="$1"
  local shuffle_flag="$2"  # "" or "--shuffle-words"

  echo "=========================================="
  echo "LAPE Shuffling Pipeline - ${mode_label}"
  echo "Model: $MODEL_ID"
  echo "Shuffle flag: ${shuffle_flag:-<none>}"
  echo "Output root: $OUT_DIR"
  echo "=========================================="

  echo "[1/4] Collecting activation counts (${mode_label})..."
  python3 "$ROOT/scripts/activations_count.py" "$MODEL_ID" \
    --hidden-dim "$HIDDEN_DIM" \
    --dataset-configs "${COUNT_CONFIGS[@]}" \
    --layer "$ACT_LAYERS" \
    --out-dir "$OUT_DIR" \
    --out-path "$ACTS_SUBDIR" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    $shuffle_flag

  echo "[2/4] Identifying language-specific neurons (${mode_label})..."
  python3 "$ROOT/scripts/identify.py" \
    --model "$MODEL_ID" \
    --layer "$IDENTIFY_LAYERS" \
    --dataset-configs "${IDENTIFY_CONFIGS[@]}" \
    --in-dir "$OUT_DIR" \
    --in-path "$ACTS_SUBDIR" \
    --out-dir "$OUT_DIR" \
    --out-path "$SPEC_SUBDIR" \
    --out-filename "lape_neuron.pt" \
    --algorithm "lape" \
    --lang-specific \
    $shuffle_flag

  echo "[3/4] Identifying shared neurons (${mode_label})..."
  for shared_count in {2..15}; do
    python3 "$ROOT/scripts/identify.py" \
      --model "$MODEL_ID" \
      --layer "$IDENTIFY_LAYERS" \
      --dataset-configs "${IDENTIFY_CONFIGS[@]}" \
      --in-dir "$OUT_DIR" \
      --in-path "$ACTS_SUBDIR" \
      --out-dir "$OUT_DIR" \
      --out-path "$SHARED_SUBDIR" \
      --out-filename "lape_shared_${shared_count}.pt" \
      --algorithm "lape" \
      --lang-shared \
      --shared-count "$shared_count" \
      $shuffle_flag
  done

  echo "[4/4] Completed ${mode_label} run."
  echo
}

run_mode "Original (no shuffling)" ""
run_mode "Shuffled words" "--shuffle-words"

echo "All LAPE shuffling runs completed. Artifacts stored under $OUT_DIR"

