#!/bin/bash

input_directory=/raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted/

# Iterate over all JSONL files with "XXX" in their name
for file in "$input_directory"*test*.jsonl; do
    # Skip if no matching file
    [ -e "$file" ] || continue

    # Count total number of lines
    total_lines=$(wc -l < "$file")

    # Calculate 10% of total lines (rounded up)
    ten_percent=$(( (total_lines + 9) / 10 ))

    # Create the 10% file (overwrite original)
    head -n "$ten_percent" "$file" > "${file}.tmp"

    # Create the rest file
    tail -n +"$((ten_percent + 1))" "$file" > "${file%.jsonl}_rest.jsonl"

    # Replace the original file with the 10% version
    mv "${file}.tmp" "$file"

    echo "Split $file → $file (10%) and ${file%.jsonl}_rest.jsonl (90%)"
done
