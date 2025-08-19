#!/bin/sh

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun \
    --rdzv-endpoint localhost:29504 \
    --nnodes 1 \
    --nproc_per_node 6 \
    $(which modalities) run --config_file_path /home/behzad_shomali/modalities/config_files/training/config_lorem_ipsum_long_fsdp2_2B.yaml