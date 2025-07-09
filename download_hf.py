from datasets import load_dataset, DownloadConfig
import datasets, aiohttp

import os
import time

# Set environment variables for better timeout handling
os.environ['HF_DATASETS_TIMEOUT'] = '7200'  # 2 hour timeout
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')

ds_name = "openslr/librispeech_asr"

ds = load_dataset(ds_name, split="train.clean.100", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
print("Download completed successfully!")

# # Retry logic with exponential backoff
# max_retries = 5
# base_delay = 30  # seconds

# for attempt in range(max_retries):
#     try:
#         print(f"Attempt {attempt + 1}/{max_retries}: Downloading {ds_name}...")
#         download_config = DownloadConfig(
#             max_retries=5, 
#             resume_download=True,
#             force_download=False  # Use cached files if available
#         )
            

#         ds = load_dataset(ds_name, split="train", storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
#         print("Download completed successfully!")
#         break
#     except Exception as e:
#         print(f"Attempt {attempt + 1} failed: {e}")
#         if attempt < max_retries - 1:
#             delay = base_delay * (2 ** attempt)  # Exponential backoff
#             print(f"Retrying in {delay} seconds...")
#             time.sleep(delay)
#         else:
#             print("All attempts failed. The partial download should be cached.")
#             print("You can try running the script again to resume from where it left off.")
#             raise

# # print where the dataset is saved
# print(f"Dataset '{ds_name}' downloaded and saved to: {ds.cache_files[0]['filename']}")