#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/identify.py" \
    --model 'meta-llama/Llama-3.2-1B' \
    --layer model.layers.{0..15}.mlp \
    --dataset-configs 'facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}' 'google-research-datasets/paws-x:{en,de,fr,es,ja,ko,zh}' 'openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}' \
    --in-path 'mlp_acts_count/meta-llama/Llama-3.2-1B' \
    --out-path 'mlp_acts_specific/meta-llama/Llama-3.2-1B' \
    --algorithm 'lape' \
    --out-filename 'lape_neuron.pt'
