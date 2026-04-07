for layer_num in {0..15}; do
    CUDA_VISIBLE_DEVICES=2 python run_probing_cv.py \
    --model_path "${MI_MODELS_DIR:-/home/models}/meta-llama_Llama-3.2-1B" \
    --model_name "Llama-3.2-1B" \
    --sae-model "${MI_MODELS_DIR:-/home/models}/sae-Llama-3.2-1B-131k/" \
    --layers "layers.${layer_num}.mlp" \
    --langs en de fr it pt hi es ru tr ja ko zh ur bn\
    --exp flores_plus-shared-flores_plus \
    --features  'fam' 'geo' 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
                    'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
                    'inventory_phoible_aa' 'inventory_phoible_gm' \
                    'inventory_phoible_saphon' 'inventory_phoible_spa' \
                    'inventory_phoible_ph' 'inventory_phoible_ra' \
                    'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
                    'inventory_knn' 'syntax_average' 'phonology_average' \
                    'inventory_average'\
    --split dev \
    --batch-size 16 \
    --device cuda \
    --all-neurons \
    --raw-model
done
 