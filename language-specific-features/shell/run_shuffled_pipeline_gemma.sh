#!/usr/bin/env bash
# Combined pipeline for Gemma-2-2b: run once with original prompts and once
# with shuffled prompts. Each run executes the standard flow:
#   activations_to_sae_features -> sae_features_count -> identify (all + shared)

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$DIR/.."

MODEL="${MI_MODELS_DIR:-/home/models}/gemma-2-2b"
HF_MODEL_ID="google/gemma-2-2b"
# Keep directory naming consistent with activations_to_sae_features output
MODEL_DIR="$(basename "$MODEL")"
SAE_MODEL="gemma-scope-2b-pt-mlp-canonical"
OUT_DIR="$ROOT/output_shuffling_gemma"

LAYERS="model.layers.{0..25}.mlp"

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

FEATURE_DIR="$OUT_DIR/sae_features/$MODEL_DIR/$SAE_MODEL"
COUNT_DIR="$OUT_DIR/sae_features_count/$MODEL_DIR/$SAE_MODEL"
SPECIFIC_DIR="$OUT_DIR/sae_features_specific/$MODEL_DIR/$SAE_MODEL"
SHARED_DIR="$OUT_DIR/sae_features_shared/$MODEL_DIR/$SAE_MODEL"

run_pipeline_mode() {
  local mode_label="$1"
  local shuffle_flag="${2:-}"
  local -a extra_args=()
  local -a cmd count_cmd identify_cmd shared_cmd

  if [[ -n "$shuffle_flag" ]]; then
    extra_args+=("$shuffle_flag")
  fi

  echo "=========================================="
  echo "Gemma pipeline run: ${mode_label}"
  echo "Shuffle flag: ${shuffle_flag:-<none>}"
  echo "=========================================="

  echo "[1/4] Collecting SAE features (${mode_label})..."

  echo "  - ${XNLI_DATASET}"
  cmd=(python3 "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$XNLI_DATASET"
    --split train
    --lang "${XNLI_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 1000
    --sae-model "$SAE_MODEL"
    --batch 500
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "  - ${PAWSX_DATASET}"
  cmd=(python3 "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$PAWSX_DATASET"
    --split train
    --lang "${PAWSX_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 1000
    --sae-model "$SAE_MODEL"
    --batch 500
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "  - ${FLORES_DATASET}"
  cmd=(python3 "$ROOT/scripts/activations_to_sae_features.py" "$MODEL" "$FLORES_DATASET"
    --split dev
    --lang "${FLORES_LANGS[@]}"
    --layer "$LAYERS"
    --start 0 --end 997
    --sae-model "$SAE_MODEL"
    --batch 500
    --out-dir "$OUT_DIR")
  cmd+=("${extra_args[@]}")
  "${cmd[@]}"

  echo "[2/4] Computing SAE statistics (${mode_label})..."

  echo "  - ${XNLI_DATASET}"
  stats_cmd=(python3 "$ROOT/scripts/sae_statistics.py" "$MODEL" "$XNLI_DATASET"
    --lang "${XNLI_LANGS[@]}"
    --layer "$LAYERS"
    --sae-model "$SAE_MODEL"
    --in-dir "$OUT_DIR"
    --out-dir "$OUT_DIR")
  stats_cmd+=("${extra_args[@]}")
  "${stats_cmd[@]}"

  echo "  - ${PAWSX_DATASET}"
  stats_cmd=(python3 "$ROOT/scripts/sae_statistics.py" "$MODEL" "$PAWSX_DATASET"
    --lang "${PAWSX_LANGS[@]}"
    --layer "$LAYERS"
    --sae-model "$SAE_MODEL"
    --in-dir "$OUT_DIR"
    --out-dir "$OUT_DIR")
  stats_cmd+=("${extra_args[@]}")
  "${stats_cmd[@]}"

  echo "  - ${FLORES_DATASET}"
  stats_cmd=(python3 "$ROOT/scripts/sae_statistics.py" "$MODEL" "$FLORES_DATASET"
    --lang "${FLORES_LANGS[@]}"
    --layer "$LAYERS"
    --sae-model "$SAE_MODEL"
    --in-dir "$OUT_DIR"
    --out-dir "$OUT_DIR")
  stats_cmd+=("${extra_args[@]}")
  "${stats_cmd[@]}"

  echo "[3/4] Counting SAE feature activations (${mode_label})..."
  count_cmd=(python3 "$ROOT/scripts/sae_features_count.py"
    --output-type "EncoderOutput"
    --hidden-dim 65536
    --dataset-configs "${DATASET_CONFIGS[@]}"
    --layer "$LAYERS"
    --in-path "$FEATURE_DIR"
    --out-path "$COUNT_DIR")
  count_cmd+=("${extra_args[@]}")
  "${count_cmd[@]}"

  echo "[4/4] Running identify (LAPE) (${mode_label})..."
  identify_cmd=(python3 "$ROOT/scripts/identify.py"
    --model "$HF_MODEL_ID"
    --sae-model "$SAE_MODEL"
    --layer "$LAYERS"
    --dataset-configs "${DATASET_CONFIGS[@]}"
    --in-path "$COUNT_DIR"
    --out-path "$SPECIFIC_DIR"
    --out-filename 'lape_all.pt'
    --topk-threshold-ratio 0.5
    --example-rate 0.98
    --algorithm 'sae_lape'
    --lang-specific)
  identify_cmd+=("${extra_args[@]}")
  "${identify_cmd[@]}"

for count in {2..15}; do
    shared_cmd=(python3 "$ROOT/scripts/identify.py"
      --model "$HF_MODEL_ID"
      --sae-model "$SAE_MODEL"
      --layer "$LAYERS"
      --dataset-configs "${DATASET_CONFIGS[@]}"
      --in-path "$COUNT_DIR"
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
# run_pipeline_mode "Shuffled words" "--shuffle-words"

echo "All Gemma pipeline runs completed."

# run_pipeline_mode "Original (no shuffling)" ""
