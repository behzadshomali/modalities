CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 torchrun \
    --rdzv-endpoint localhost:29522 \
    --nnodes 1 \
    --nproc_per_node 2 $(which modalities) run \
    --config_file_path /raid/s3/opengptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/LoopMTP_paper_expert_experiments/alignment_loss_ablate/5MTP.yaml ;

sleep 1m; 

CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 torchrun \
    --rdzv-endpoint localhost:29522 \
    --nnodes 1 \
    --nproc_per_node 2 $(which modalities) run \
    --config_file_path /raid/s3/opengptx/behzad_shomali/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/LoopMTP_paper_expert_experiments/main_runs/5MTP.yaml ;

