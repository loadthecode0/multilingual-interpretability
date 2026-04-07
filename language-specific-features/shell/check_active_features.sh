#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/check_active_features.py" \
    --model meta-llama/Llama-3.2-1B \
    --layer model.layers.{0..15}.mlp \
    --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
    --batch 500 \
    --lape-result-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --classifier-type count \
    --text "Hamilton has been the primary songwriter, guitarist, and vocalist for Brothers Past, as well as co-producer for all of their recorded releases." "Olga Alexandrowna Girja (russisch Ольга Александровна Гиря; * 4. Juni 1991 in Langepas) ist eine russische Schachspielerin und seit 2009 Großmeister der Frauen (WGM)."
