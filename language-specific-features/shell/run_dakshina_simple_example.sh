#!/bin/bash

# Simple Dakshina Example Script
# Shows how to use both native and romanized versions of Dakshina dataset

echo "Dakshina Dataset Example - Native vs Romanized"
echo "=============================================="

# Configuration
SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
OUTPUT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output"
MODEL_PATH="${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B"
SAE_MODEL="EleutherAI/sae-Llama-3.2-1B-131k"
LANG="hi"  # Hindi
LAYER="layers.2.mlp"
DATASET_START=0
DATASET_END=100

# Create output directories
mkdir -p "$OUTPUT_DIR"

echo "Processing Hindi (hi) with layer $LAYER"
echo "Samples: $DATASET_START to $DATASET_END"
echo ""

# Example 1: Native Hindi text
echo "1. Processing NATIVE Hindi text..."
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --model "$MODEL_PATH" \
    --dataset-configs "dakshina:$LANG:train:$DATASET_START:$DATASET_END" \
    --layer "$LAYER" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --hidden-dim 2048 \
    --device "cuda:0"

echo "✓ Native Hindi processing completed"
echo "  Output: $OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL/dakshina/hi.pt"
echo ""

# Example 2: Romanized Hindi text
echo "2. Processing ROMANIZED Hindi text..."
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --model "$MODEL_PATH" \
    --dataset-configs "dakshina:$LANG:train:$DATASET_START:$DATASET_END" \
    --layer "$LAYER" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --hidden-dim 2048 \
    --device "cuda:0" \
    --romanized

echo "✓ Romanized Hindi processing completed"
echo "  Output: $OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL/dakshina/hi_romanized.pt"
echo ""

# Example 3: Native Hindi with word shuffling
echo "3. Processing NATIVE Hindi with word shuffling..."
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --model "$MODEL_PATH" \
    --dataset-configs "dakshina:$LANG:train:$DATASET_START:$DATASET_END" \
    --layer "$LAYER" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --hidden-dim 2048 \
    --device "cuda:0" \
    --shuffle-words

echo "✓ Native Hindi with shuffling completed"
echo "  Output: $OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL/dakshina/hi_shuffled.pt"
echo ""

# Example 4: Romanized Hindi with word shuffling
echo "4. Processing ROMANIZED Hindi with word shuffling..."
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --model "$MODEL_PATH" \
    --dataset-configs "dakshina:$LANG:train:$DATASET_START:$DATASET_END" \
    --layer "$LAYER" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/$SAE_MODEL" \
    --hidden-dim 2048 \
    --device "cuda:0" \
    --romanized \
    --shuffle-words

echo "✓ Romanized Hindi with shuffling completed"
echo "  Output: $OUTPUT_DIR/sae_features_count/Llama-3.2-1B/$SAE_MODEL/dakshina/hi_romanized_shuffled.pt"
echo ""

echo "=============================================="
echo "All examples completed!"
echo "=============================================="
echo ""
echo "File naming convention:"
echo "  - {lang}.pt                    (native)"
echo "  - {lang}_romanized.pt          (romanized)"
echo "  - {lang}_shuffled.pt           (native + shuffled)"
echo "  - {lang}_romanized_shuffled.pt (romanized + shuffled)"
echo ""
echo "Check the debug output to see the difference between native and romanized text!"


