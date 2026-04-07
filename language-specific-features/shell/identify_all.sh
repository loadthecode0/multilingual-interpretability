#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$DIR/identify_neuron_lape.sh" >logs/identify_neuron_lape.txt
bash "$DIR/identify_sae_lape_top_1_per_layer_by_entropy.sh" >logs/identify_sae_lape_top_1_per_layer_by_entropy.txt
bash "$DIR/identify_sae_lape_top_1_per_layer_by_freq.sh" >logs/identify_sae_lape_top_1_per_layer_by_freq.txt
bash "$DIR/identify_sae_lape_top_10_by_freq.sh" >logs/identify_sae_lape_top_10_by_freq.txt
bash "$DIR/identify_sae_lape_top_10_by_entropy.sh" >logs/identify_sae_lape_top_10_by_entropy.txt
bash "$DIR/identify_sae_lape_all.sh" >logs/identify_sae_lape_all.txt
