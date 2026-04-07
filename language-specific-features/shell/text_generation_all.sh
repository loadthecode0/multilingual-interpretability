#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0 \
    --max-new-token 256 \
    --lang en \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.2 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.3 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.4 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.5 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.6 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.8 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.2 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.3 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.4 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.5 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.6 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..14}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier 0.8 \
    --max-new-token 256 \
    --lang deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.2 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.3 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.4 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.5 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.6 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -0.8 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0
    
python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -1 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -1.2 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -1.4 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -1.5 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0

python "$DIR/../scripts/text_generation.py" 'meta-llama/Llama-3.2-1B' \
    --layer 'model.layers.{0..15}.mlp' \
    --in-path 'sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_all.pt' \
    --out-path 'text_generation/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/all' \
    --multiplier -2 \
    --max-new-token 256 \
    --lang eng_Latn \
    --times 100 \
    --seed 0
