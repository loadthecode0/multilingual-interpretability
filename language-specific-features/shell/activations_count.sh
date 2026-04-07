#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

# Extract activations
python "$DIR/../scripts/activations_count.py" meta-llama/Llama-3.2-1B \
    --hidden-dim 8192 \
    --dataset-configs 'facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}:train:0:1000' 'google-research-datasets/paws-x:{en,de,fr,es,ja,ko,zh}:train:0:1000' 'openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}:dev:0:1000' \
    --layer "model.layers.{0..15}.mlp.act_fn" \
    --out-path mlp_acts_count/meta-llama/Llama-3.2-1B
