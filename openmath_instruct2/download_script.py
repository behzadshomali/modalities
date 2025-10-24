import json
import requests
from huggingface_hub import list_repo_files, hf_hub_download

# login to HF if needed
from huggingface_hub import login
TOKEN = None
login(token=TOKEN)


def estimate_tokens(text):
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def download_openmath_data(train_target_tokens, test_target_tokens, output_dir):
    """
    Download data from nvidia/OpenMathInstruct-2 up to target token count
    
    Args:
        target_tokens: Approximate number of tokens to download
        output_file: Output JSONL file path
    """
    repo_id = "nvidia/OpenMathInstruct-2"
    
    print(f"Fetching files from {repo_id}...")
    # files = list_repo_files(repo_id)
    
    # # Find parquet files
    # parquet_files = [f for f in files if f.endswith('.parquet')]
    
    # if not parquet_files:
    #     print("No parquet files found. Trying direct dataset loading...")
    from datasets import load_dataset
    dataset = load_dataset(repo_id, split="train", streaming=True)
    return _download_from_streaming(dataset, train_target_tokens, test_target_tokens, output_dir)
    

def _download_from_streaming(dataset, train_target_tokens, test_target_tokens, output_dir):
    """Fallback for streaming datasets"""
    total_tokens = 0
    samples_written = 0
    
    with open(output_dir + "openmath_train_data.jsonl", 'w', encoding='utf-8') as f:
        for sample in dataset:
            text = json.dumps(sample)
            tokens = estimate_tokens(text)
            
            if total_tokens + tokens > train_target_tokens and samples_written > 0:
                break
            
            f.write(json.dumps(sample) + '\n')
            total_tokens += tokens
            samples_written += 1
            
            if samples_written % 100 == 0:
                print(f"Downloaded {samples_written} samples (~{total_tokens:,} tokens)")
    with open(output_dir + "openmath_test_data.jsonl", 'w', encoding='utf-8') as f:
        total_tokens = 0
        samples_written = 0
        for sample in dataset:
            text = json.dumps(sample)
            tokens = estimate_tokens(text)
            
            if total_tokens + tokens > test_target_tokens and samples_written > 0:
                break
            
            f.write(json.dumps(sample) + '\n')
            total_tokens += tokens
            samples_written += 1
            
            if samples_written % 100 == 0:
                print(f"Downloaded {samples_written} test samples (~{total_tokens:,} tokens)")
    
    print(f"\nCompleted!")
    print(f"Total samples: {samples_written}")
    print(f"Estimated tokens: {total_tokens:,}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    # Example: download ~5M tokens worth of data
    train_target_tokens = 5_000_000
    test_target_tokens = train_target_tokens * 0.05
    download_openmath_data(train_target_tokens=train_target_tokens, test_target_tokens=test_target_tokens, output_dir="/raid/s3/opengptx/behzad_shomali/data/")