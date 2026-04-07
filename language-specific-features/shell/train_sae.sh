#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/train_sae.py" meta-llama/Llama-3.2-1B EleutherAI/rpj-v2-sample \
        --hf_token {HF_TOKEN} \
        --expansion_factor 64 \
        --normalize_decoder True \
        --k 32 \
        --multi_topk False \
        --skip_connection False \
        --batch_size 4 \
        --grad_acc_steps 2 \
        --micro_acc_steps 1 \
        --lr_warmup_steps 1000 \
        --auxk_alpha 0.0 \
        --dead_feature_threshold 10000000 \
        --hookpoints layers.0.mlp \
        --init_seeds 0 \
        --layer_stride 1 \
        --transcode False \
        --distribute_modules False \
        --save_every 1000 \
        --log_to_wandb True \
        --run_name 'sae' \
        --wandb_log_frequency 1 \
        --split train \
        --ctx_len 2048 \
        --load_in_8bit False \
        --resume False \
        --text_column raw_content \
        --shuffle_seed 42 \
        --data_preprocessing_num_proc 112