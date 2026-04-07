#!/bin/bash

# Dakshina Pipeline Script
# Runs the complete pipeline for Dakshina dataset with both native and romanized versions
# Usage: ./run_dakshina_pipeline.sh

set -e  # Exit on any error

echo "=========================================="
echo "Dakshina Dataset Pipeline"
echo "=========================================="

# Configuration
SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
OUTPUT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output"
MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
SAE_MODEL="EleutherAI/sae-Llama-3.2-1B-131k"
DATASET="dakshina"
LAYERS=("model.layers.{0..15}.mlp")
LANGUAGES=("hi" "bn" "ur" "sd")  # Hindi, Bengali, Urdu, Sindhi
DATASET_START=0
DATASET_END=1000
BATCH_SIZE=100
DEVICE="cuda:0"
HF_MODEL_ID="meta-llama/Llama-3.2-1B"
LOCAL_SAE_DIR="${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k"

# Create output directories
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  SAE Model: $SAE_MODEL"
echo "  Dataset: $DATASET"
echo "  Languages: ${LANGUAGES[*]}"
echo "  Layers: ${LAYERS[*]}"
echo "  Samples: $DATASET_START to $DATASET_END"
echo "  Device: $DEVICE"
echo ""

# Function to run pipeline for a specific configuration
run_pipeline() {
    local lang=$1
    local romanized=$2
    local layer=$3
    
    echo "=========================================="
    echo "Processing: $lang ($([ "$romanized" = "true" ] && echo "romanized" || echo "native")) - $layer"
    echo "=========================================="
    
    # Step 1: Collect activations count
    echo "Step 1: Collecting activations count..."
    python3 "$SCRIPT_DIR/scripts/activations_count.py" \
        --model "$MODEL_PATH" \
        --dataset-configs "$DATASET:$lang:train:$DATASET_START:$DATASET_END" \
        --layer "$layer" \
        --out-dir "$OUTPUT_DIR" \
        --out-path "sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
        --hidden-dim 2048 \
        --device "$DEVICE" \
        $([ "$romanized" = "true" ] && echo "--romanized" || echo "") \
        --shuffle-words false
    
    Step 2: Collect SAE features
    echo "Step 2: Collecting SAE features..."
    python3 "$SCRIPT_DIR/scripts/activations_to_sae_features.py" \
        "$MODEL_PATH" \
        "$DATASET" \
        --lang "hi" "bn" "ur" "sd" \
        --layer "$LAYERS" \
        --start "$DATASET_START" \
        --end "$DATASET_END" \
        --split "train" \
        --sae-model "$SAE_MODEL" \
        --local-sae-dir "$LOCAL_SAE_DIR" \
        --batch "$BATCH_SIZE" \
        --out-dir "$OUTPUT_DIR" \
        $([ "$romanized" = "true" ] && echo "--romanized" || echo "") 
    
    Step 3: Generate SAE statistics
    echo "Step 3: Generating SAE statistics..."
    python3 "$SCRIPT_DIR/scripts/sae_statistics.py" \
        "$HF_MODEL_ID" \
        "$DATASET" \
        --lang "hi" "bn" "ur" "sd" \
        --layer "$LAYERS" \
        --sae-model "$SAE_MODEL" \
        --in-dir "$OUTPUT_DIR" \
        --out-dir "$OUTPUT_DIR" \
        $([ "$romanized" = "true" ] && echo "--romanized" || echo "") 
    
    echo "Step 3: Counting SAE feature activations..."
    python3 "$SCRIPT_DIR/scripts/sae_features_count.py" \
        --output-type "EncoderOutput" \
        --hidden-dim 131072 \
        --dataset-configs "$DATASET:{hi,bn,ur,sd}" \
        --layer "$LAYERS" \
        --in-path "./output/sae_features/Llama-3.2-1B/$SAE_MODEL" \
        --out-path "./output/sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
        $([ "$romanized" = "true" ] && echo "--romanized" || echo "") 
    
    # Step 4: Identify language-specific features
    echo "Step 4: Identifying language-specific features..."
    for count in {2..15}; do
        python3 "$SCRIPT_DIR/scripts/identify.py" \
            --model "$HF_MODEL_ID" \
            --sae-model "$SAE_MODEL" \
            --layer "model.layers.{0..15}.mlp" \
            --dataset-configs "$DATASET:{hi,bn,ur,sd}" \
            --in-path './output/sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
            --out-path './output/sae_features_shared/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
            --out-filename "lape_shared_${count}.pt" \
            --topk-threshold-ratio 0.5 \
            --example-rate 0.98 \
            --lang-shared \
            --shared-count "$count" \
            --algorithm "sae_lape" \
            $([ "$romanized" = "true" ] && echo "--romanized" || echo "") 
        done
        
    echo "✓ Completed: $lang ($([ "$romanized" = "true" ] && echo "romanized" || echo "native")) - $layer"
    echo ""
}

# Main execution
echo "Starting Dakshina pipeline execution..."
echo ""

# Run pipeline for each language, both native and romanized, for each layer
# for lang in "${LANGUAGES[@]}"; do
# for layer in "${LAYERS[@]}"; do
# Run native version
# run_pipeline "$LANGUAGES" "false" "$LAYERS"

# Run romanized version
run_pipeline "$LANGUAGES" "true" "$LAYERS"
    # done
# done

echo "=========================================="
echo "Dakshina Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  sae_features_count/Llama-3.2-1B/$SAE_MODEL/$DATASET/"
echo "    - {lang}.pt (native)"
echo "    - {lang}_romanized.pt (romanized)"
echo ""
echo "  sae_features/Llama-3.2-1B/$SAE_MODEL/$DATASET/"
echo "    - {lang}/ (native)"
echo "    - {lang}_romanized/ (romanized)"
echo ""
echo "  statistics/Llama-3.2-1B/$SAE_MODEL/$DATASET/"
echo "    - summary/{layer}/ (native)"
echo "    - summary_romanized/{layer}/ (romanized)"
echo "    - dataset_token_activations/{layer}/ (native)"
echo "    - dataset_token_activations_romanized/{layer}/ (romanized)"
echo ""
echo "  lape_all/Llama-3.2-1B/$SAE_MODEL/$DATASET/"
echo "    - lape_all_{lang}_{layer}.pt (native)"
echo "    - lape_all_{lang}_{layer}_romanized.pt (romanized)"
