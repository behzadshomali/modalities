#!/usr/bin/env bash
set -euo pipefail

# Input and output directories
input_path_dir="/raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts"
output_dir="/raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted"

# Ratios
train_ratio=0.9
val_ratio=0.05

mkdir -p "$output_dir"

for input_path in "$input_path_dir"/*.jsonl; do
    [ -f "$input_path" ] || continue
    echo "Processing $(basename "$input_path")..."

    total_lines=$(wc -l < "$input_path")
    train_end=$(printf "%.0f" "$(echo "$total_lines * $train_ratio" | bc -l)")
    val_end=$(printf "%.0f" "$(echo "$train_end + ($total_lines * $val_ratio)" | bc -l)")

    base_name=$(basename "$input_path")
    base_name="${base_name%.jsonl}"  # Remove .jsonl extension
    train_file="$output_dir/${base_name}_train.jsonl"
    val_file="$output_dir/${base_name}_val.jsonl"
    test_file="$output_dir/${base_name}_test.jsonl"

    awk -v train_end="$train_end" -v val_end="$val_end" -v tf="$train_file" -v vf="$val_file" -v tf2="$test_file" '
        NR <= train_end { print > tf; next }
        NR <= val_end   { print > vf; next }
                        { print > tf2 }
    ' "$input_path"

    echo "Done! Train: $train_end, Val: $((val_end - train_end)), Test: $((total_lines - val_end)))"
done
