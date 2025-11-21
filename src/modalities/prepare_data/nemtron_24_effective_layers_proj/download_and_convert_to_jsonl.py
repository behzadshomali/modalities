import os
from datasets import load_dataset
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


def download_arrow_files(
    target_dir,
    dataset_name,
    split,
    subset
):
    dataset = load_dataset(
        dataset_name, 
        split=split,
        name=subset,
        cache_dir=target_dir
    )

    print("Dataset downloaded succesfully!")
    return dataset

def convert_to_jsonl(
    dataset,
    output_file,
    chunks_num=1,
    batch_size=10000
):
    output_file = Path(output_file)
    dataset_len = len(dataset)

    # shuffle dataset
    dataset = dataset.shuffle(seed=2000)

    if chunks_num == 1:
        with open(output_file, "w") as f:
            for start_idx in tqdm(range(0, dataset_len, batch_size), desc=f"Writing {output_file}"):
                batch = dataset[start_idx:start_idx+batch_size]
                for record in batch:
                    f.write(json.dumps(record) + "\n")
    else:
        # Calculate chunk boundaries
        chunk_size = (dataset_len + chunks_num - 1) // chunks_num  # ceil division
        for i in range(chunks_num):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, dataset_len)
            chunk_file = output_file.with_name(f"{output_file.stem}_part{i+1}{output_file.suffix}")

            with open(chunk_file, "w") as f:
                for batch_start in tqdm(range(start_idx, end_idx, batch_size),
                                        desc=f"Writing {chunk_file}"):
                    batch_end = min(batch_start + batch_size, end_idx)
                    indices = list(range(batch_start, batch_end))
                    batch = dataset.select(indices)
                    for record in batch:
                        f.write(json.dumps(record) + "\n")
    
    print("Converting data from arrow to jsonl was successfully done!")
    


if __name__ == "__main__":
    arrow_dir = "/raid/s3/opengptx/behzad_shomali/data"
    dataset_name = "nvidia/Nemotron-CC-Math-v1"
    split = "train"
    subset = "4plus"
    output_file = f"{arrow_dir}/{dataset_name.split('/')[-1]}_{split}_{subset}.jsonl"

    dataset = download_arrow_files(arrow_dir, dataset_name, split, subset)
    convert_to_jsonl(dataset, output_file, chunks_num=10)


