#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/train_sae_pretokenize.py" meta-llama/Llama-3.2-1B EleutherAI/rpj-v2-sample \
        --batch_size 10000 \
        --split train \
        --ctx_len 2048 \
        --hf_token {HF_TOKEN} \
        --text_column raw_content \
        --data_preprocessing_num_proc 112 \
        --output_dir pretokenized \
        --hf_username {HF_USER} \
        --hf_dataset_name rpj-v2-sample-pretokenized
