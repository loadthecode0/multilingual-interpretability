#!/bin/bash

# Example script showing how to use Dakshina dataset with the language-specific-features pipeline
# This script demonstrates running the pipeline for Hindi (hi) and Bengali (bn) from Dakshina dataset

echo "Running Dakshina dataset pipeline example..."
echo "=========================================="

# Set up paths
SCRIPT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features"
OUTPUT_DIR="${MI_PROJECT_ROOT:-$(pwd)}/language-specific-features/output"

# Create output directories
mkdir -p "$OUTPUT_DIR"

# Example 1: Run activations count for Hindi
echo "Step 1: Running activations count for Hindi (Dakshina)"
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --dataset-configs "dakshina:hi:train" \
    --dataset-start 0 \
    --dataset-end 100 \
    --text-column "text" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k" \
    --model-path "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" \
    --sae-model "EleutherAI/sae-Llama-3.2-1B-131k" \
    --layers "layers.2.mlp" \
    --device "cuda:0"

# Example 2: Run activations count for Bengali  
echo "Step 2: Running activations count for Bengali (Dakshina)"
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --dataset-configs "dakshina:bn:train" \
    --dataset-start 0 \
    --dataset-end 100 \
    --text-column "text" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k" \
    --model-path "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" \
    --sae-model "EleutherAI/sae-Llama-3.2-1B-131k" \
    --layers "layers.2.mlp" \
    --device "cuda:0"

# Example 3: Run with shuffling enabled
echo "Step 3: Running activations count for Hindi with word shuffling"
python3 "$SCRIPT_DIR/scripts/activations_count.py" \
    --dataset-configs "dakshina:hi:train" \
    --dataset-start 0 \
    --dataset-end 100 \
    --text-column "text" \
    --out-dir "$OUTPUT_DIR" \
    --out-path "sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k" \
    --model-path "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" \
    --sae-model "EleutherAI/sae-Llama-3.2-1B-131k" \
    --layers "layers.2.mlp" \
    --device "cuda:0" \
    --shuffle-words

echo "Dakshina pipeline example completed!"
echo "Check the output directory for results: $OUTPUT_DIR"
