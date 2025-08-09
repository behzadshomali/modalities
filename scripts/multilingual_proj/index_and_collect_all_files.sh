#!/bin/bash

languages=("en" "fra_Latn" "deu_Latn" "ita_Latn" "spa_Latn")


for language in "${languages[@]}"; do
    echo "Processing language: $language"

    modalities data create_raw_index \
    --index_path /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/mem_map/multilingual_fineweb_${language}_train.idx \
        /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted/multilingual_fineweb_${language}_train.jsonl

    modalities data create_raw_index \
    --index_path /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/mem_map/multilingual_fineweb_${language}_test.idx \
        /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted/multilingual_fineweb_${language}_test.jsonl

    modalities data create_raw_index \
    --index_path /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/mem_map/multilingual_fineweb_${language}_val.idx \
        /raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted/multilingual_fineweb_${language}_val.jsonl
done

# echo "Indexing completed for all languages."

for language in "${languages[@]}"; do
    echo "Packing data for language: $language"

    modalities data pack_encoded_data modalities/src/modalities/multilingual_proj/configs/multilingual_fineweb_${language}_train_dataset_config.yaml
    
    modalities data pack_encoded_data modalities/src/modalities/multilingual_proj/configs/multilingual_fineweb_${language}_test_dataset_config.yaml

    modalities data pack_encoded_data modalities/src/modalities/multilingual_proj/configs/multilingual_fineweb_${language}_val_dataset_config.yaml
done