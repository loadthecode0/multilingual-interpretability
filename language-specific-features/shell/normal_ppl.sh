#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../scripts/ppl.py" meta-llama/Llama-3.2-1B openlanguagedata/flores_plus \
       --split devtest \
       --lang eng_Latn deu_Latn fra_Latn ita_Latn por_Latn hin_Deva spa_Latn tha_Thai bul_Cyrl rus_Cyrl tur_Latn vie_Latn jpn_Jpan kor_Hang cmn_Hans \
       --start 0 --end 1000 \
       --out-path normal/ppl.pt
