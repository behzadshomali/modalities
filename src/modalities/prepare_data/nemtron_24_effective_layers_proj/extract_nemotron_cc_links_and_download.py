import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_paths(file_path):
    """
    Reads the file containing paths and filters for specific Nemotron criteria.
    """
    extracted_links = []
    total_links = 0
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, "r") as f:
        for line in f:
            clean_line = line.strip()
            if "quality=high/kind=actual/kind2=actual" in clean_line:
                extracted_links.append(clean_line)
            total_links += 1

    print(f"Processing complete: Kept {len(extracted_links)}/{total_links} links.")
    return extracted_links

def create_complete_links(extracted_paths):
    """
    Prepends the CommonCrawl base URL to the relative paths.
    """
    prefix = "https://data.commoncrawl.org/"
    completed_links = []
    for path in extracted_paths:
        if path.startswith("/"):
            path = path[1:]
        completed_links.append(prefix + path)

    return completed_links

def download_and_process_file(url, output_dir):
    """
    1. Downloads file via WGET.
    2. Decompresses via ZSTD.
    3. Removes the .zstd file to save space.
    """
    time.sleep(0.005)
    try:
        filename = os.path.basename(url)
        file_path = os.path.join(output_dir, filename)
        final_jsonl_path = file_path.replace(".zstd", "")

        # # Skip if the DECOMPRESSED file already exists (save time on re-runs)
        # if os.path.exists(final_jsonl_path):
        #     return f"Skipped (Already exists): {filename}"

        # --- Step 1: Download ---
        # -nc: No Clobber (don't re-download if .zstd exists)
        # -q: Quiet
        # -P: Output directory
        download_cmd = ["wget", "-nc", "-q", "-P", output_dir, url]
        subprocess.run(download_cmd, check=True)

        # --- Step 2 & 3: Decompress and Remove ---
        # -d: Decompress
        # --rm: Remove the source (.zstd) file after successful decompression
        # -f: Force overwrite if output exists
        zstd_cmd = ["zstd", "-d", "--rm", "-f", file_path]
        subprocess.run(zstd_cmd, check=True)
        
        return f"Success: {filename} -> Decompressed & Cleaned"

    except subprocess.CalledProcessError as e:
        return f"Failed (Process Error): {url}"
    except Exception as e:
        return f"Error ({e}): {url}"

def parallel_manager(link_list, output_dir, max_workers=8):
    """
    Manages the parallel execution.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting processing of {len(link_list)} files with {max_workers} workers...")
    print(f"Target Directory: {output_dir}")
    print("-" * 50)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_and_process_file, url, output_dir): url for url in link_list}
        
        completed_count = 0
        for future in as_completed(future_to_url):
            completed_count += 1
            result = future.result()
            
            # Print specific errors, but keep successes quiet to reduce noise
            if "Failed" in result or "Error" in result:
                print(f"[{completed_count}/{len(link_list)}] {result}")
            elif completed_count % 10 == 0:
                print(f"[{completed_count}/{len(link_list)}] Progress update...")

            

    end_time = time.time()
    print("-" * 50)
    print(f"Job finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    all_links_path = "/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/all_links_jsonl_path.txt"
    output_directory = "/raid/s3/opengptx/behzad_shomali/data/nvidia___nemotron-cc_JSONL/" 
    
    # Increase workers if you have high bandwidth and fast CPU
    PARALLEL_WORKERS = 16
    
    # --- EXECUTION ---
    extracted_paths = extract_paths(all_links_path)
    
    if extracted_paths:
        full_urls = create_complete_links(extracted_paths[2400:])
        parallel_manager(full_urls, output_directory, max_workers=PARALLEL_WORKERS)
    else:
        print("No matching links found.")