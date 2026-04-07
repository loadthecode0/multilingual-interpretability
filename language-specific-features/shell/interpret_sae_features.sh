#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/interpret.py" meta-llama/Llama-3.2-1B EleutherAI/sae-Llama-3.2-1B-131k \
    --hookpoints layers.0.mlp layers.1.mlp layers.2.mlp layers.3.mlp layers.4.mlp layers.5.mlp layers.6.mlp layers.7.mlp layers.8.mlp layers.9.mlp layers.10.mlp layers.11.mlp layers.12.mlp layers.13.mlp layers.14.mlp layers.15.mlp \
    --explainer_model openai/gpt-4.1 \
    --explainer_model_max_len 5120 \
    --explainer_provider openrouter \
    --name interpret_sae_features \
    --seed 42 \
    --num_examples_per_scorer_prompt 5 \
    --dataset_configs facebook/xnli:{en,de,fr,hi,es,th,bg,ru,tr,vi}:train google-research-datasets/paws-x:{en,de,fr,es}:train openlanguagedata/flores_plus:{eng_Latn,deu_Latn,fra_Latn,ita_Latn,por_Latn,hin_Deva,spa_Latn,tha_Thai,bul_Cyrl,rus_Cyrl,tur_Latn,vie_Latn,jpn_Jpan,kor_Hang,cmn_Hans}:dev \
    --dataset_start 0 --dataset_end 1000 \
    --dataset_repo "" \
    --dataset_split "" \
    --dataset_name "" \
    --batch_size 32 \
    --cache_ctx_len 256 \
    --n_tokens 10_000_000 \
    --n_splits 5 \
    --faiss_embedding_cache_enabled false \
    --example_ctx_len 32 \
    --min_examples 200 \
    --n_non_activating 50 \
    --non_activating_source random \
    --center_examples true \
    --n_examples_train 40 \
    --n_examples_test 50 \
    --n_quantiles 10 \
    --train_type top \
    --test_type quantiles \
    --lape_result_path sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt \
    --env local \
    --verbose false \
    --log_results true \
    --populate_cache true \
    --scoring true
