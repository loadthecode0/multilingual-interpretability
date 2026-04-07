#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/sae_statistics.py" "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" facebook/xnli \
    --lang en de fr hi es th bg ru tr vi \
    --layer model.layers.{0..15}.mlp \
    --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
    --in-dir ./output \
    --out-dir ./output \

python "$DIR/../scripts/sae_statistics.py" "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" google-research-datasets/paws-x \
    --lang en de fr es ja ko zh \
    --layer model.layers.{0..15}.mlp \
    --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
    --in-dir ./output \
    --out-dir ./output \

python "$DIR/../scripts/sae_statistics.py" "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" openlanguagedata/flores_plus \
    --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --layer model.layers.{0..15}.mlp \
    --sae-model EleutherAI/sae-Llama-3.2-1B-131k \
    --in-dir ./output \
    --out-dir ./output \
