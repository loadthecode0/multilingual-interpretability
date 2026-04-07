#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/sae_features_count.py" \
    --output-type "EncoderOutput" \
    --hidden-dim 131072 \
    --dataset-configs 'facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}' 'google-research-datasets/paws-x:{en,de,fr,es,ja,ko,zh}' 'openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path './output/sae_features/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
    --out-path './output/sae_features_count/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k' \
    "$@"
