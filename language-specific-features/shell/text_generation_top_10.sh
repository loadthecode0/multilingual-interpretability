#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 0 \
    --max-new-token 40 \
    --lang en \
    --times 10 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..13}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 0.5 \
    --max-new-token 40 \
    --lang fr hi es th bg ru tr vi \
    --times 10 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..13}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 1 \
    --max-new-token 40 \
    --lang fr hi es th bg ru tr vi \
    --times 10 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..13}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 1.25 \
    --max-new-token 40 \
    --lang fr hi es th bg ru tr vi \
    --times 10 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..13}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 1.5 \
    --max-new-token 40 \
    --lang fr hi es th bg ru tr vi \
    --times 10 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..13}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/top_10' \
    --multiplier 2 \
    --max-new-token 40 \
    --lang fr hi es th bg ru tr vi \
    --times 10 \
    --seed 0
