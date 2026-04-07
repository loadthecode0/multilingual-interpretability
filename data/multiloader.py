import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Add parent directory to sys.path to find utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import paths

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDatasetType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
import random


class MultilingualDatasetManager:
    """Manages multiple parallel multilingual datasets with unified language encoding."""
    
    def __init__(self, 
                 storage_dir: str = os.path.join(paths.PROJECT_ROOT, "data/multilingual_datasets"),
                 config_path: str = os.path.join(paths.PROJECT_ROOT, "data/multilingual_datasets/data_config.json"),
                 model_name: str = "bert-base-multilingual-cased",
                 max_length: int = 128,
                 verbose: bool = False):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose
        
        # Load configurations from JSON
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            self.language_mappings = config_data['language_mappings']
            self.dataset_configs = config_data['datasets']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cached_datasets = {}
    
    def _log(self, message: str):
        """Helper method for conditional logging"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def _get_cache_path(self, dataset_name: str, language: str, split: str) -> Path:
        """Return cache path in format: storage_dir/dataset_name/language-split.pkl"""
        dataset_dir = self.storage_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / f"{language}-{split}.pkl"

    def _get_dataset_language_code(self, dataset_name: str, common_lang: str) -> str:
        """Convert common language code to dataset-specific code"""
        if common_lang not in self.language_mappings:
            raise ValueError(f"Language '{common_lang}' not supported. Available: {list(self.language_mappings.keys())}")
        if dataset_name not in self.language_mappings[common_lang]:
            raise ValueError(f"Language '{common_lang}' not supported for dataset '{dataset_name}'")
        return self.language_mappings[common_lang][dataset_name]
    
    def _load_dataset_split(self, dataset_name: str, common_lang: str, split: str) -> Optional[HFDatasetType]:
        """Load a single dataset split with proper language handling"""
        config = self.dataset_configs[dataset_name]
        dataset_lang_code = None
        if config['language_param_type'] != "parallel_subset":
            dataset_lang_code = self._get_dataset_language_code(dataset_name, common_lang)
        
        
        if config['language_param_type'] == 'flores_subset':
            dataset = load_dataset(config['name'], "all", split=split)
            import pandas as pd
            df = pd.DataFrame({"sentence": list(dataset[f"sentence_{dataset_lang_code}"])})
            dataset = HFDatasetType.from_pandas(df)

        elif config['language_param_type'] == 'flores_plus_subset':
            print(f"Loading dataset {dataset_lang_code} for {split} for flores_plus_subset")
            # dataset = load_dataset(config['name'], dataset_lang_code)
            ds = load_dataset(
                config['name'],
                dataset_lang_code,
                split=split
            )

            # Convert streaming → in-memory HF dataset
            # (flores+ subsets ~2k rows → safe)
            ds = list(ds)  # materialize
            from datasets import Dataset
            dataset = Dataset.from_list(ds)

        elif config['language_param_type'] == 'filter':
            dataset = load_dataset(config['name'], split=split)
            if 'language' in dataset.column_names:
                dataset = dataset.filter(lambda x: x['language'] == dataset_lang_code)
            elif 'lang' in dataset.column_names:
                dataset = dataset.filter(lambda x: x['lang'] == dataset_lang_code)
                
        elif config['language_param_type'] == 'pair':
            if common_lang == 'en':
                available_pairs = config.get('language_pairs', [])
                en_pairs = [pair for pair in available_pairs if pair.startswith('en-')]
                if en_pairs:
                    dataset = load_dataset(config['name'], en_pairs[0], split=split)
                else:
                    return None
            else:
                pair = f"en-{common_lang}"
                if pair in config.get('language_pairs', []):
                    dataset = load_dataset(config['name'], pair, split=split)
                else:
                    return None

        elif config['language_param_type'] == 'parallel_subset':
            # For datasets like JW300 / Europarl (subset = en-xx, split=train)
            if common_lang == 'en':
                # Pick first non-English supported lang to construct a subset
                non_en = config['supported_languages'][1]
                subset_name = f"en-{non_en}"
                dataset = load_dataset(config['name'], subset_name, split=split)
                col = "english"
            else:
                subset_name = f"en-{common_lang}"
                dataset = load_dataset(config['name'], subset_name, split=split)
                col = "non_english"
            
            # Select up to 1K examples
            dataset = dataset.select(range(min(1000, len(dataset))))

            # Normalize into "sentence" column
            import pandas as pd
            df = pd.DataFrame({"sentence": list(dataset[col])})
            dataset = HFDatasetType.from_pandas(df)

        elif dataset_name == "dakshina":
            base_dir = Path(paths.DAKSHINA_DIR)
            file_path = base_dir / common_lang / "romanized" / f"{common_lang}.romanized.rejoined.tsv"
            import pandas as pd
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                names=["native", "romanized"],
                engine="python",         # more forgiving than C parser
                quoting=3,               # ignore quotes
                on_bad_lines="skip"      # skip malformed rows
                )
            print(df.head())
            dataset = HFDatasetType.from_pandas(df)
            # return dataset

        else:
            dataset = load_dataset(config['name'], split=split)

        
            
        return dataset
            
        # except Exception as e:
        #     if self.verbose:
        #         print(f"[ERROR] Failed to load {dataset_name}-{common_lang}-{split}: {e}")
        #     return None

    def download_and_cache_dataset(self, 
                                 dataset_name: str, 
                                 languages: Optional[List[str]] = None,
                                 splits: Optional[List[str]] = None,
                                 force_redownload: bool = False) -> Dict[str, Dict[str, HFDatasetType]]:
        """Download and cache a dataset for multiple languages using common language codes"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        languages = languages or config['supported_languages']
        splits = splits or config['split_configs']
        
        if dataset_name not in self.cached_datasets:
            self.cached_datasets[dataset_name] = {}
        
        downloaded_data = {}
        
        for common_lang in languages:
            if common_lang not in config['supported_languages']:
                self._log(f"Language {common_lang} not supported for {dataset_name}, skipping")
                continue
                
            lang_data = {}
            for split in splits:
                cache_path = self._get_cache_path(dataset_name, common_lang, split)
                
                if cache_path.exists() and not force_redownload:
                    self._log(f"Loading cached {dataset_name}-{common_lang}-{split}")
                    with open(cache_path, 'rb') as f:
                        dataset = pickle.load(f)
                        lang_data[split] = dataset
                    continue
                
                self._log(f"Downloading {dataset_name}-{common_lang}-{split}")
                dataset = self._load_dataset_split(dataset_name, common_lang, split)
                
                if dataset is not None:
                    lang_data[split] = dataset
                    self._log(f"Successfully loaded {len(dataset)} examples for {common_lang}-{split}")
                    
                    for ex in dataset.select(range(min(3, len(dataset)))):
                        self._log(f"Example: {ex}")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(dataset, f)
            
            if lang_data:
                self.cached_datasets[dataset_name][common_lang] = lang_data
                downloaded_data[common_lang] = lang_data
        
        return downloaded_data

    def download_all_datasets(self, 
                            languages: Optional[List[str]] = None,
                            force_redownload: bool = False) -> Dict[str, Dict[str, Dict[str, HFDatasetType]]]:
        """Download all configured datasets using common language codes"""
        all_data = {}
        for dataset_name in self.dataset_configs.keys():
            self._log(f"Processing dataset: {dataset_name}")
            all_data[dataset_name] = self.download_and_cache_dataset(
                dataset_name, languages, force_redownload=force_redownload
            )
        return all_data
    
    def get_dataset(self, dataset_name: str, language: str, split: str) -> Optional[HFDatasetType]:
        """Get a specific cached dataset using common language code"""
        try:
            print(f"Getting dataset: {dataset_name}-{language}-{split}")
            return self.cached_datasets[dataset_name][language][split]
        except KeyError:
            print(f"KeyError: {dataset_name}-{language}-{split}")
            cache_path = self._get_cache_path(dataset_name, language, split)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    dataset = pickle.load(f)
                    if dataset_name not in self.cached_datasets:
                        self.cached_datasets[dataset_name] = {}
                    if language not in self.cached_datasets[dataset_name]:
                        self.cached_datasets[dataset_name][language] = {}
                    self.cached_datasets[dataset_name][language][split] = dataset
                    return dataset
            return None
    
    def create_dataloader(self, 
                         dataset_name: str, 
                         language: str, 
                         split: str,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0,
                         shuffle_words: bool = False,
                         romanized: bool = False,
                         debug: bool = False) -> Optional[DataLoader]:
        """Create a PyTorch DataLoader using common language code
        
        Args:
            shuffle_words: If True, shuffle words inside each sentence before tokenization
        """
        hf_dataset = self.get_dataset(dataset_name, language, split)
        print(f"Loaded {len(hf_dataset)} examples for {language}-{split}")
        if hf_dataset is None:
            return None

        hf_dataset = hf_dataset.select(range(min(1024, len(hf_dataset))))
        
        text_field = self.dataset_configs[dataset_name]['text_fields'][0]

        if romanized:
            text_field = 'romanized'

        pytorch_dataset = TokenizedDataset(
            hf_dataset, self.tokenizer, text_field, self.max_length, shuffle_words=shuffle_words, debug=debug
        )
        collator = DataCollatorWithPadding(self.tokenizer)

        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator
        )

        print(f"batch size: {batch_size}, len dataloader: {len(dataloader)}")

        return dataloader
    
    def get_available_languages(self, dataset_name: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """Get available languages using common language codes"""
        if dataset_name:
            return self.dataset_configs[dataset_name]['supported_languages']
        else:
            return {name: config['supported_languages'] for name, config in self.dataset_configs.items()}
    
    def get_available_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Get summary of available cached data using common language codes"""
        summary = {}
        for dataset_name in self.cached_datasets:
            summary[dataset_name] = {}
            for language in self.cached_datasets[dataset_name]:
                summary[dataset_name][language] = list(self.cached_datasets[dataset_name][language].keys())
        return summary
    
    def cleanup_cache(self, dataset_name: Optional[str] = None):
        """Clean up cached data"""
        if dataset_name:
            pattern = f"{dataset_name}_*.pkl"
        else:
            pattern = "*.pkl"
        
        for path in self.storage_dir.glob(pattern):
            path.unlink()
            
    def get_parallel_dataloaders(self, 
                               dataset_name: str,
                               languages: List[str],
                               split: str,
                               batch_size: int = 32,
                               num_workers: int = 0) -> Dict[str, DataLoader]:
        """Create parallel dataloaders for multiple languages - useful for activation analysis"""
        dataloaders = {}
        for lang in languages:
            dl = self.create_dataloader(dataset_name, lang, split, batch_size, False, num_workers)
            if dl is not None:
                dataloaders[lang] = dl
        return dataloaders


class TokenizedDataset(Dataset):
    """PyTorch Dataset for tokenized text data"""
    
    def __init__(self, hf_dataset: HFDatasetType, tokenizer, text_field: str, max_length: int, shuffle_words: bool = False, debug: bool = False):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        self.shuffle_words = shuffle_words
        self.debug = debug
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        text = self.hf_dataset[idx][self.text_field]
        # if idx < 3:
        # print(idx, text)
        if text is None:
            # fallback if dataset has 'sentence' instead
            text = self.hf_dataset[idx].get("native", None)
            # print(text)
        if self.shuffle_words:
            if self.debug:
                print("[DEBUG] Shuffling words to test word order significance.")
            words = text.split()
            random.shuffle(words)
            text = " ".join(words)
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}
