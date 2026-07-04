# Multilingual Language Models Encode Script Over Linguistic Structure

Code for the paper **"Multilingual Language Models Encode Script Over Linguistic Structure"** (ACL 2026, Main Conference) — [arXiv:2604.05090](https://arxiv.org/abs/2604.05090).

We study how multilingual LMs internally organise representations across languages, and find that they organise primarily around **orthography / surface form (script)** rather than abstract linguistic identity, with typological structure only becoming linearly accessible in deeper layers. The analysis combines: language-associated unit discovery via the **LAPE** metric, **sparse autoencoder (SAE)** feature decomposition, **romanization** and **word-order shuffling** perturbations, layer-wise **probing** against typological (lang2vec) features, and **causal interventions** on the identified units.

Models studied: **Llama-3.2-1B** (`meta-llama/Llama-3.2-1B`) and **Gemma-2-2B** (`google/gemma-2-2b`), with their pretrained MLP SAEs (`EleutherAI/sae-Llama-3.2-1B-131k` and `gemma-scope-2b-pt-mlp-canonical`).

---

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Dataset loading (`multiloader.py`, `data_config.json`) and FLORES+ romanization (`romanize_flores_plus.py`). Cached FLORES+ dev splits ship under `data/multilingual_datasets/flores_plus/`. |
| `models/` | Model / SAE loaders (`loader.py`, `gemmascope.py`). |
| `utils/` | Shared config (`config.py`) and path/env resolution (`paths.py`). |
| `language-specific-features/` | Core module: SAE training, activation collection, **LAPE** identification, feature interpretation, intervention/perplexity, steered generation, language classification. Python in `scripts/`, launchers in `shell/`. |
| `probing/` | Layer-wise probing of activations against lang2vec typological features, plus dominance ranking. |
| `causal_intervention/` | Ablation experiments (zero / mean / cross-language) on identified units, with significance testing against random-neuron controls. |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (the analysis scripts default to `cuda`). Key libraries: `torch`, `transformers`, `nnsight` (activation hooks), `sae-lens` + `eai-sparsify` (SAE loading/training), `lang2vec` (typological features), `fasttext` (language-ID baseline), `delphi`-based auto-interpretation (vendored under `language-specific-features/scripts/delphi/`).

### 2. Configure paths and authentication

Model and project locations are resolved from environment variables (`utils/paths.py`). Set these before running anything:

```bash
export MI_MODELS_DIR=/path/to/models        # where local checkpoints live (default: /home/models)
export MI_PROJECT_ROOT=/path/to/multilingual-interpretability   # default: auto-detected repo root
export HF_TOKEN=hf_...                       # required for gated models + dataset downloads
```

`MI_MODELS_DIR` is expected to contain the local checkpoints referenced throughout:

```
$MI_MODELS_DIR/meta-llama_Llama-3.2-1B
$MI_MODELS_DIR/gemma-2-2b
$MI_MODELS_DIR/sae-Llama-3.2-1B-131k       # EleutherAI Llama SAE
```

Llama and Gemma are gated on the Hugging Face Hub — request access on their model pages and log in (`huggingface-cli login` or `HF_TOKEN`) before downloading. The Gemma SAE (`gemma-scope-2b-pt-mlp-canonical`) is pulled from the Hub at runtime.

> The `shell/` scripts hard-code `CUDA_VISIBLE_DEVICES` and specific language/layer sweeps. Treat them as the canonical recipes for each experiment and adjust device IDs, `--out-dir`, and language lists to your setup.

---

## Data preparation

Multilingual data is loaded via `data/multiloader.py`, which caches per-language splits as pickles under `data/multilingual_datasets/<dataset>/<lang>-<split>.pkl` (FLORES+ dev splits are already included). Configured datasets (`data/multilingual_datasets/data_config.json`): FLORES / FLORES+, WMT19, OPUS-100, JW300, Europarl, and Dakshina (local romanized Indic TSVs).

For the **romanization** experiment, generate romanized FLORES+ text (transliterates non-Latin scripts to Roman script via PyICU):

```bash
# ./data/run_romanize_flores_plus.sh [SPLIT] [OUTPUT_DIR] [--ascii]
./data/run_romanize_flores_plus.sh dev ./data/multilingual_datasets/flores_plus_romanized
```

This writes per-language `*.dev.romanized.jsonl` plus a combined file to the output directory. Add `--ascii` to strip diacritics. The Dakshina Indic corpus must be downloaded separately into `data/multilingual_datasets/dakshina/`.

---

## Workflow

All commands below are run from the repo root unless noted. Each stage writes artifacts consumed by the next; the shell launchers in each module chain the exact language/layer sweeps used in the paper.

### A. Language-specific features (LAPE + SAE) — `language-specific-features/`

The core pipeline runs one of two tracks that share the LAPE identification step.

**Track A — SAE features (main result).** Collect SAE feature activations → aggregate counts → identify language-specific/shared features via LAPE:

```bash
cd language-specific-features

# 1. Collect per-example SAE feature activations (XNLI + PAWS-X + FLORES+)
bash shell/collect_sae_features.sh

# 2. Aggregate to per-language / per-layer counts
bash shell/sae_features_count.sh

# 3. LAPE on SAE feature space — language-specific and shared feature sets
bash shell/identify_sae_lape_all.sh        # -> lape_all.pt
bash shell/identify_sae_lape_shared.sh     # -> lape_shared_{2..15}.pt
bash shell/identify_all.sh                 # runs the full identify sweep (top-k / per-layer variants)
```

**Track B — raw MLP neurons (baseline).** Collect neuron activation counts → original LAPE:

```bash
python scripts/activations_count.py meta-llama/Llama-3.2-1B \
    --hidden-dim 8192 \
    --dataset-configs "openlanguagedata/flores_plus:{eng_Latn,deu_Latn,hin_Deva,rus_Cyrl,jpn_Jpan,cmn_Hans}:dev:0:1000" \
    --layer "model.layers.{0..15}.mlp.act_fn" \
    --out-dir ./output --out-path mlp_acts_count/Llama-3.2-1B \
    --local-model-path "$MI_MODELS_DIR/meta-llama_Llama-3.2-1B"

bash shell/identify_neuron_lape.sh          # -> lape_neuron.pt
```

**(Optional) Train an SAE from scratch** instead of using the pretrained ones:

```bash
bash shell/train_sae_pretokenize.sh   # pre-tokenize the training corpus (optional)
bash shell/train_sae.sh               # train MLP SAE on Llama-3.2-1B
```

**Interpret** identified features with an LLM (via OpenRouter; set the provider key it expects):

```bash
bash shell/interpret_sae_features.sh
```

**Downstream evaluation** of the identified units:

```bash
bash shell/normal_ppl.sh                        # baseline perplexity (no intervention)
bash shell/sae_features_intervene_ppl_all.sh    # perplexity under SAE-feature steering
bash shell/neuron_intervene_ppl.sh              # perplexity under neuron ablation
bash shell/text_generation_all.sh               # steered free-text generation
bash shell/classify.sh                          # language-ID probe (SAE / neuron / fasttext)
```

### B. Perturbation experiments (script vs. structure)

End-to-end pipelines that re-run the SAE/neuron LAPE tracks under each perturbation and diff the resulting unit sets. `--romanization` isolates the effect of script; word-shuffling isolates word order.

```bash
cd language-specific-features

# Romanization (script perturbation) — Llama and Gemma
bash shell/run_icu_dakshina_pipeline.sh --romanized
bash shell/run_icu_dakshina_pipeline_gemma.sh --romanized
bash shell/run_icu_dakshina_lape_pipeline.sh          # raw-neuron variant

# Word-order shuffling (structure perturbation)
bash shell/run_shuffled_pipeline.sh
bash shell/run_shuffled_pipeline_gemma.sh
bash shell/run_shuffled_lape_pipeline.sh              # raw-neuron variant

# Build normal-vs-perturbed neuron overlap lists for causal follow-up
python scripts/generate_neuron_lists.py               # writes neuron_lists/{romanization,shuffling}/
```

### C. Probing typological structure across layers — `probing/`

Probe layer activations against lang2vec typological features to see where syntax/phonology/inventory becomes linearly decodable.

```bash
cd probing

# 0. Fetch lang2vec typological feature matrices
python get_l2v_features.py \
    --config_path ../data/multilingual_datasets/data_config.json \
    --out_csv lang2vec_probing/features/lang2vec_combined.csv

# 1. Ridge-regression CV probing, per layer (edit CUDA_VISIBLE_DEVICES inside)
bash run_llama_probe.sh      # Llama-3.2-1B, layers 0-15
bash run_gemma_probe.sh      # Gemma-2-2b,  layers 0-25

# 2. Aggregate + rank
python aggregate_probing_results.py                    # per-layer summaries (edit `mode` = llama/gemma)
python build_agg_scores.py --base-dir lang2vec_probing --scores-subdir results_cv --output-subdir agg_scores_cv
python compute_dominance.py --base-dir lang2vec_probing \
    --scores-subdir results_cv --output-subdir dominance/results_cv --topk 200
```

`compute_dominance.py` produces the `probing/dominance/**` CSVs that the causal-intervention scripts consume.

### D. Causal interventions — `causal_intervention/`

Ablate the top-ranked (or perturbation-selected) units and measure KL divergence and perplexity change against size-matched random-neuron controls.

```bash
# Dominance-based ablation (uses probing/dominance/** from stage C)
bash causal_intervention/shell/run_zero_ablation_sae.sh          # Llama SAE, zero ablation
bash causal_intervention/shell/run_mean_ablation_sae_cross_lang.sh  # cross-language mean ablation
bash causal_intervention/shell/run_gemma_zero_ablation_sae.sh    # Gemma

# Perturbation neuron-list ablation (uses neuron_lists/** from stage B)
bash causal_intervention/shell_neurons_lists/run_shuffling_zero_llama_raw.sh
bash causal_intervention/shell_neurons_lists/run_romanization_gemma_raw.sh
bash causal_intervention/shell_neurons_lists/run_random_mean.sh romanization en hi 50 200  # random control

# Significance testing / summary
python causal_intervention/summarize_results.py \
    --lang all --model Llama-3.2-1B --experiment shuffling \
    --input_root causal_results_neuron_lists_100ex
```

---

## Notes

- Scripts accept `bracex`-style layer expansions, e.g. `--layer model.layers.{0..15}.mlp`.
- Llama uses 16 MLP layers (hidden dim 8192, SAE width 131072); Gemma uses 26 layers (hidden dim 9216, SAE width 65536). The Gemma pipelines set `--hidden-dim` and layer ranges accordingly.
- LAPE feature/neuron sets are saved as `.pt` files under each stage's `--out-path`; `pt_to_csv*.py` and `lape_pt_to_csv.py` convert them to CSV for inspection.

## Citation

```bibtex
@inproceedings{verma2026multilingual,
  title     = {Multilingual Language Models Encode Script Over Linguistic Structure},
  author    = {Verma, Aastha A K and Chatterjee, Anwoy and Gupta, Mehak and Chakraborty, Tanmoy},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```
