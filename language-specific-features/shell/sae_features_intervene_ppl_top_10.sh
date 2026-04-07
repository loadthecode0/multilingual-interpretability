#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

langs=(
       "eng_Latn",
       "deu_Latn",
       "fra_Latn",
       "ita_Latn",
       "por_Latn",
       "hin_Deva",
       "spa_Latn",
       "tha_Thai",
       "bul_Cyrl",
       "rus_Cyrl",
       "tur_Latn",
       "vie_Latn",
       "jpn_Jpan",
       "kor_Hang",
       "cmn_Hans",
)

# Define multipliers
multipliers=(0.2 -0.2)

# Function to run Python script with given parameters
run_intervention() {
       local multiplier_value=$1
       local lang=$2

       python "$DIR/../scripts/ppl.py" meta-llama/Llama-3.2-1B openlanguagedata/flores_plus \
              --split devtest \
              --layer model.layers.{0..15}.mlp \
              --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
              --start 0 --end 1000 \
              --intervention-type sae-features \
              --intervention-lang "$lang" \
              --multiplier "${multiplier_value}" \
              --lape-result-path sae_features_specific/meta-llama/Llama-3.2-1B/EleutherAI/sae-Llama-3.2-1B-131k/lape_top_10.pt \
              --out-path sae_intervention/top_10/max/mult_${multiplier_value}/ppl_${lang}.pt \
              --lape-value-type final_indice_global_max_active
}

# Run all combinations
for multiplier_value in "${multipliers[@]}"; do
       for lang in "${langs[@]}"; do
              run_intervention "$multiplier_value" "$lang"
       done
done
