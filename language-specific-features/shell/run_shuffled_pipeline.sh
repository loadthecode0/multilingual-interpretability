#!/usr/bin/env bash
# Combined pipeline runner: executes the end-to-end flow once with the
# original prompts and once with shuffled prompts. The shuffled run simply
# adds --shuffle-words so that every downstream script reads the shuffled
# inputs that get written to disk.

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$DIR/.."

MODEL="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
HF_MODEL_ID="meta-llama/Llama-3.2-1B"
SAE_MODEL="EleutherAI/sae-Llama-3.2-1B-131k"
LOCAL_SAE_DIR="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"
OUT_DIR="$ROOT/output"
FEATURE_COUNT_DIR="$OUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL"
SPECIFIC_DIR="$OUT_DIR/sae_features_specific/Llama-3.2-1B/$SAE_MODEL"
SHARED_DIR="$OUT_DIR/sae_features_shared/Llama-3.2-1B/$SAE_MODEL"

LAYERS="model.layers.{0..15}.mlp"

# Datasets and languages (mirrors existing shell scripts)
XNLI_DATASET="facebook/xnli"
XNLI_LANGS=(en de fr hi es th bg ru tr vi)

PAWSX_DATASET="google-research-datasets/paws-x"
PAWSX_LANGS=(en de fr es ja ko zh)

FLORES_DATASET="openlanguagedata/flores_plus"
FLORES_LANGS=(eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans)

DATASET_CONFIGS=(
  "$XNLI_DATASET:{en,de,fr,hi,es,th,bg,ru,tr,vi}"
  "$PAWSX_DATASET:{en,de,fr,es,ja,ko,zh}"
  "$FLORES_DATASET:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}"
)

run_pipeline_mode() {
  local mode_label="$1"
  local shuffle_flag="${2:-}"
  local -a extra_args=()
  local -a cmd count_cmd identify_cmd shared_cmd

  if [[ -n "$shuffle_flag" ]]; then
    extra_args+=("$shuffle_flag")
  fi

  echo "=========================================="
  echo "Combined pipeline run: ${mode_label}"
  echo "Shuffle flag: ${shuffle_flag:-<none>}"
  echo "=========================================="

  echo "[1/4] Collecting SAE features (${mode_label})..."

  echo "  - ${XNLI_DATASET}"
  cmd=(python "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$XNLI_DATASET"
    --split train
    --lang "${XNLI_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 1000
    --sae-model "$SAE_MODEL"
    --batch 500
    --local-sae-dir "$LOCAL_SAE_DIR"
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "  - ${PAWSX_DATASET}"
  cmd=(python "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$PAWSX_DATASET"
    --split train
    --lang "${PAWSX_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 1000
    --sae-model "$SAE_MODEL"
    --batch 500
    --local-sae-dir "$LOCAL_SAE_DIR"
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "  - ${FLORES_DATASET}"
  cmd=(python "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$FLORES_DATASET"
    --split dev
    --lang "${FLORES_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 997
    --sae-model "$SAE_MODEL"
    --batch 500
    --local-sae-dir "$LOCAL_SAE_DIR"
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "[2/4] (Optional) SAE statistics step skipped (unchanged)."

  echo "[3/4] Counting SAE features (${mode_label})..."
  count_cmd=(bash "$DIR/sae_features_count.sh")
  count_cmd+=("${extra_args[@]}")
  "${count_cmd[@]}"

  echo "[4/4] Running identify (LAPE) (${mode_label})..."
  identify_cmd=(python "$ROOT/scripts/identify.py"
    --model "$HF_MODEL_ID"
    --sae-model "$SAE_MODEL"
    --layer "$LAYERS"
    --dataset-configs "${DATASET_CONFIGS[@]}"
    --in-path "$FEATURE_COUNT_DIR"
    --out-path "$SPECIFIC_DIR"
    --out-filename 'lape_all.pt'
    --topk-threshold-ratio 0.5
    --example-rate 0.98
    --algorithm 'sae_lape'
    --lang-specific)
  identify_cmd+=("${extra_args[@]}")
  "${identify_cmd[@]}"

  for count in {2..15}; do
    shared_cmd=(python "$ROOT/scripts/identify.py"
      --model "$HF_MODEL_ID"
      --sae-model "$SAE_MODEL"
      --layer "$LAYERS"
      --dataset-configs "${DATASET_CONFIGS[@]}"
      --in-path "$FEATURE_COUNT_DIR"
      --out-path "$SHARED_DIR"
      --out-filename "lape_shared_${count}.pt"
      --topk-threshold-ratio 0.5
      --example-rate 0.98
      --algorithm 'sae_lape'
      --lang-shared
      --shared-count "$count")
    shared_cmd+=("${extra_args[@]}")
    "${shared_cmd[@]}"
  done

  echo "Done. ${mode_label} run finished."
  echo
}

run_pipeline_mode "Original (no shuffling)" ""
run_pipeline_mode "Shuffled words" "--shuffle-words"

echo "All pipeline runs completed."





