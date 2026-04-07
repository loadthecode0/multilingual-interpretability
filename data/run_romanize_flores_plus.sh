#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_romanize_flores_plus.sh [SPLIT] [OUTPUT_DIR] [--ascii]
# Examples:
#   ./run_romanize_flores_plus.sh dev
#   ./run_romanize_flores_plus.sh dev ./results
#   ./run_romanize_flores_plus.sh devtest ./results --ascii
#
# Languages requested:
#   hindi, marathi, bengali, urdu, russian, bulgarian, japanese, chinese, korean, english, spanish
# FLORES-PLUS config codes:
#   hin_Deva, mar_Deva, ben_Beng, urd_Arab, rus_Cyrl, bul_Cyrl, jpn_Jpan, cmn_Hans, kor_Hang, eng_Latn, spa_Latn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT="${1:-dev}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/results_non_ascii}"
ASCII_FLAG="${3:-}"

LANGS="hin_Deva,mar_Deva,ben_Beng,urd_Arab,rus_Cyrl,bul_Cyrl,jpn_Jpan,cmn_Hans,kor_Hang,eng_Latn,spa_Latn"

python "$SCRIPT_DIR/romanize_flores_plus.py" \
	--languages "$LANGS" \
	--split "$SPLIT" \
	--output-dir "$OUTPUT_DIR" \
	${ASCII_FLAG:+--ascii}


