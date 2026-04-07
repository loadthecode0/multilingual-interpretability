#!/usr/bin/env bash
set -euo pipefail

# Full romanized cross-analysis pipeline:
# 1. Runs the standard romanized pipeline into a dedicated directory
# 2. Re-runs sae_statistics.py with cross filtering using the native outputs
# 3. Regenerates cross_analysis CSVs

SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
NATIVE_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output_romanization_expt"
CROSS_ROOT="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output_romanization_expt_cross"

mkdir -p "$CROSS_ROOT"

echo "[1/4] Running romanized pipeline (outputs stored in $NATIVE_ROOT)..."
# bash "$SCRIPT_DIR/shell/run_icu_dakshina_pipeline.sh" --romanized

echo "[2/4] Copying romanized outputs into $CROSS_ROOT ..."
rsync -a --delete "$NATIVE_ROOT/" "$CROSS_ROOT/"

echo "[3/4] Filtering romanized stats to native-selected features..."
PYTHONPATH="$SCRIPT_DIR/scripts" python "$SCRIPT_DIR/scripts/sae_statistics.py" \
  meta-llama/Llama-3.2-1B icu_dakshina \
  --lang hi mr bn ur ru bg ja zh ko en es \
  --layer "model.layers.{0..15}.mlp" \
  --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
  --in-dir "$CROSS_ROOT" \
  --out-dir "$CROSS_ROOT" \
  --romanized \
  --cross-romanized \
  --native-output-root "$NATIVE_ROOT"

echo "[4/4] Producing cross-analysis CSVs..."
PYTHONPATH="$SCRIPT_DIR/scripts" python "$SCRIPT_DIR/scripts/cross_analysis.py" \
  --output-root "$CROSS_ROOT" \
  --cross-dir cross_analysis


