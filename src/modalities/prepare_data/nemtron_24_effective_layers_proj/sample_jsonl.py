import json
import csv
from pathlib import Path
from collections import defaultdict
import random
import os

def load_statistics(stats_file):
    """Load statistics from CSV file"""
    year_stats = defaultdict(lambda: {'files': [], 'total_tokens': 0, 'total_lines': 0})
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row['year'])
            filename = row['filename']
            tokens = int(row.get('total_tokens', 0))
            line_count = int(row.get('line_count', 0))
            
            year_stats[year]['files'].append({
                'filename': filename,
                'tokens': tokens,
                'line_count': line_count
            })
            year_stats[year]['total_tokens'] += tokens
            year_stats[year]['total_lines'] += line_count
    
    return year_stats

def calculate_year_proportions(year_stats, target_tokens=270_000_000_000):
    """Calculate how many tokens to sample from each year"""
    total_tokens = sum(stats['total_tokens'] for stats in year_stats.values())
    if total_tokens == 0:
        raise ValueError("Total tokens in all years is zero in statistics file.")
    
    target_per_year = {}
    for year, stats in year_stats.items():
        proportion = stats['total_tokens'] / total_tokens
        target_per_year[year] = int(round(target_tokens * proportion))
    
    # Adjust rounding so sum equals target_tokens
    sum_target = sum(target_per_year.values())
    diff = target_tokens - sum_target
    if diff != 0:
        # add/subtract 1 from the largest years until fixed (deterministic order)
        years_sorted = sorted(target_per_year.keys(), key=lambda y: year_stats[y]['total_tokens'], reverse=True)
        idx = 0
        while diff != 0:
            year = years_sorted[idx % len(years_sorted)]
            target_per_year[year] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            idx += 1
    
    return target_per_year, total_tokens

def estimate_tokens(text):
    """Estimate token count as chars / 4 (same heuristic as original)"""
    return len(text) // 4

def sample_proportionally_streaming(stats_file, data_dir, output_dir,
                                    target_tokens=270_000_000_000,
                                    num_output_files=5, seed=42):
    """
    Streaming proportional random sampler (scalable):
    - Uses adaptive per-line acceptance probabilities so the expected token count per year
      matches the target specified by the CSV proportions.
    - Streams input and writes directly to output files in round-robin; memory stays bounded.
    """
    random.seed(seed)

    # --- load stats and compute per-year targets ---
    print("Loading statistics...")
    year_stats = load_statistics(stats_file)

    target_per_year, total_tokens = calculate_year_proportions(year_stats, target_tokens)

    print(f"\nTotal tokens in corpus (CSV): {total_tokens:,}")
    print(f"Target tokens to sample (global): {target_tokens:,}")
    print("\nSampling plan by year:")
    for year in sorted(target_per_year.keys()):
        proportion = target_per_year[year] / target_tokens * 100
        print(f"  {year}: {target_per_year[year]:,} tokens ({proportion:.2f}%) (available: {year_stats[year]['total_tokens']:,})")

    # shuffle files within each year to avoid file-order bias
    print("\nShuffling file order within each year...")
    for year in year_stats:
        random.shuffle(year_stats[year]['files'])

    # prepare output files (open handles) — write streaming, round-robin
    os.makedirs(output_dir, exist_ok=True)
    output_handles = []
    for i in range(num_output_files):
        out_path = os.path.join(output_dir, f"Nemotron-CC-highQuality-sampled_data_part_{i+1}.jsonl")
        fh = open(out_path, 'w', encoding='utf-8')
        output_handles.append(fh)
    out_idx = 0  # round-robin index across output files

    # sampling bookkeeping
    tokens_sampled_per_year = defaultdict(int)
    samples_written = 0

    # iterate years (sorted for determinism)
    for year in sorted(year_stats.keys()):
        target = target_per_year.get(year, 0)
        available_tokens = year_stats[year]['total_tokens']
        if target <= 0 or available_tokens <= 0:
            print(f"\nSkipping year {year}: target {target:,}, available {available_tokens:,}")
            continue

        print(f"\nProcessing year {year} (target: {target:,} tokens, available: {available_tokens:,} tokens)...")

        tokens_collected = 0
        tokens_remaining_in_year = available_tokens  # total tokens remaining as we stream through files

        # iterate files for this year
        for file_info in year_stats[year]['files']:
            if tokens_collected >= target:
                break

            file_path = Path(data_dir) / file_info['filename']
            if not file_path.exists():
                print(f"  Warning: {file_info['filename']} not found, skipping")
                # subtract the file's recorded tokens from remaining estimate to keep adaptivity correct
                tokens_remaining_in_year = max(0, tokens_remaining_in_year - file_info.get('tokens', 0))
                continue

            # stream lines
            with open(file_path, 'r', encoding='utf-8') as inf:
                for raw in inf:
                    if tokens_collected >= target:
                        break

                    # if we've exhausted our estimate of remaining tokens (defensive)
                    if tokens_remaining_in_year <= 0:
                        # to avoid division by zero; low probability accept only if we still need tokens
                        accept_prob = 0.0
                    else:
                        # adaptive acceptance probability:
                        # expected remaining tokens to pick / tokens remaining in year
                        need = max(0, target - tokens_collected)
                        accept_prob = need / tokens_remaining_in_year
                        # clamp
                        if accept_prob > 1:
                            accept_prob = 1.0
                        elif accept_prob < 0:
                            accept_prob = 0.0

                    # small optimization: if accept_prob == 0 skip parsing JSON
                    if accept_prob <= 0:
                        # still need to decrement tokens_remaining_in_year by this line's tokens estimate,
                        # but we can estimate from the raw line text without parsing
                        line_tokens = estimate_tokens(raw)
                        tokens_remaining_in_year = max(0, tokens_remaining_in_year - line_tokens)
                        continue

                    # try parse JSON line (if invalid, skip and decrement tokens_remaining estimate)
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        line_tokens = estimate_tokens(raw)
                        tokens_remaining_in_year = max(0, tokens_remaining_in_year - line_tokens)
                        continue

                    # estimate tokens for this line
                    line_tokens = estimate_tokens(raw)

                    # decide to accept with probability accept_prob (Bernoulli trial)
                    if random.random() < accept_prob:
                        # write to output (round-robin)
                        fh = output_handles[out_idx]
                        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        out_idx = (out_idx + 1) % num_output_files

                        tokens_collected += line_tokens
                        samples_written += 1

                    # always decrement the tokens remaining estimate
                    tokens_remaining_in_year = max(0, tokens_remaining_in_year - line_tokens)

                # end file read

        tokens_sampled_per_year[year] = tokens_collected
        print(f"  Collected ~{tokens_collected:,} tokens for {year} (samples written so far: {samples_written:,})")

    # close output handles
    for fh in output_handles:
        fh.close()

    # summary
    print("\n" + "="*60)
    print("SAMPLING SUMMARY")
    print("="*60)
    total_sampled_tokens = sum(tokens_sampled_per_year.values())
    total_samples = samples_written
    print(f"Total samples written: {total_samples:,}")
    print("Tokens sampled by year:")
    for year in sorted(tokens_sampled_per_year.keys()):
        sampled = tokens_sampled_per_year[year]
        target = target_per_year.get(year, 0)
        pct = (sampled / target * 100) if target > 0 else 0.0
        print(f"  {year}: {sampled:,} / {target:,} tokens ({pct:.1f}%)")
    print(f"\nTotal tokens sampled (approx): {total_sampled_tokens:,}")
    print(f"Global target tokens: {target_tokens:,}")
    print(f"Achievement: {(total_sampled_tokens / target_tokens * 100):.1f}%")

    return {
        'tokens_sampled_per_year': dict(tokens_sampled_per_year),
        'total_tokens_sampled': total_sampled_tokens,
        'samples_written': total_samples
    }

# Example invocation (as in your original main block):
if __name__ == "__main__":
    sample_proportionally_streaming(
        stats_file='/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/file_statistics.csv',
        data_dir='/raid/s3/opengptx/behzad_shomali/data/nvidia___nemotron-cc_JSONL/',
        output_dir='/raid/s3/opengptx/behzad_shomali/data/sampled_nvidia___nemotron-cc_JSONL/',
        target_tokens=350_000_000_000,
        num_output_files=20,
        seed=42
    )
