#!/bin/bash

parts_num=({1..24})


for num in "${parts_num[@]}"; do
    echo "Processing part: $num"
    modalities data create_raw_index \
    --index_path /raid/s3/opengptx/behzad_shomali/data/sampled_nvidia___nemotron-cc_JSONL/mem_map_gpt2/nemtron_24_effective_layers_proj/Nemotron-CC-highQuality-sampled_data_part_${num}.idx \
        /raid/s3/opengptx/behzad_shomali/data/sampled_nvidia___nemotron-cc_JSONL/Nemotron-CC-highQuality-sampled_data_part_${num}.jsonl
        
    echo "Packing data for part: $num"
    modalities data pack_encoded_data /raid/s3/opengptx/behzad_shomali/modalities/config_files/data_preparation/nemtron_24_effective_layers_proj/Nemotron-CC-highQuality-sampled_data_part${num}.yaml
done

# echo "Indexing and completed for all parts."