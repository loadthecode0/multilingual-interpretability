import os
import sys
import torch
import gc


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath('.'))
from data.multiloader import MultilingualDatasetManager

def main():

    languages = ["en", "de", "fr", "it", "pt", "hi", "es", "ru", "tr", "ja", "ko", "zh", "ur", "bn"]
    
    # Initialize components
    print("\nInitializing dataset manager...", True)
    dataset_manager = MultilingualDatasetManager(
        model_name="/home/models/meta-llama_Llama-3.2-1B",
        max_length=512,
        verbose=True
    )
    

    for lang in languages:
        print(f"\n===== Processing {lang} =====", True)
        
        try:
            print(f"Downloading and caching dataset for {lang}...", True)
            dataset_manager.download_and_cache_dataset(
                "flores_plus", languages=[lang], splits=["dev"]
            )
            

            
        except Exception as e:
            print(f"Error processing language {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

main()