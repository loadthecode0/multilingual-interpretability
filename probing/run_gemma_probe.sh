for layer_num in {0..25}; do
    CUDA_VISIBLE_DEVICES=2 python3 run_probing_cv.py \
        --model_path "${MI_MODELS_DIR:-/home/models}/gemma-2-2b" \
        --model_name "Gemma-2-2b" \
        --sae-model "gemma-scope-2b-pt-mlp-canonical" \
        --layers "layers.${layer_num}.mlp" \
        --langs en de fr it pt hi es ru tr ja ko zh ur bn\
        --exp flores_plus \
        --features 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
                        'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
                        'inventory_phoible_aa' 'inventory_phoible_gm' \
                        'inventory_phoible_saphon' 'inventory_phoible_spa' \
                        'inventory_phoible_ph' 'inventory_phoible_ra' \
                        'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
                        'inventory_knn' 'syntax_average' 'phonology_average' \
                        'inventory_average' 'fam' 'geo' \
        --split dev \
        --batch-size 16 \
        --save-acts \
        --all-neurons 
        # --raw-model
done