#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

run_intervention() {
       local intervention_lang=$1
       local value=$2

       python "$DIR/../scripts/ppl.py" meta-llama/Llama-3.2-1B openlanguagedata/flores_plus \
              --split devtest \
              --layer "model.layers.{0..15}.mlp.act_fn" \
              --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
              --start 0 --end 1000 \
              --intervention-type neuron \
              --intervention-lang "$intervention_lang" \
              --value "$value" \
              --lape-result-path mlp_acts_specific/meta-llama/Llama-3.2-1B/lape_neuron.pt \
              --out-path "neuron_intervention/baseline/$([[ $value == -* ]] && echo min || echo plus)_${value#-}/ppl_${intervention_lang}.pt" \
              --neuron-intervention-method scaling \
              --lape-value-type final_indice_global_max_active
}

langs=(
       "eng_Latn"
       "deu_Latn"
       "fra_Latn"
       "ita_Latn"
       "por_Latn"
       "hin_Deva"
       "spa_Latn"
       "tha_Thai"
       "bul_Cyrl"
       "rus_Cyrl"
       "tur_Latn"
       "vie_Latn"
       "jpn_Jpan"
       "kor_Hang"
       "cmn_Hans"
)

values=(-0.2 -0.3 -0.4 0.2 0.3 0.4)

for lang in "${langs[@]}"; do
       for value in "${values[@]}"; do
              run_intervention "$lang" "$value"
       done
done
