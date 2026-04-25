#!/bin/bash
#SBATCH --job-name=train_MTP
#SBATCH --output=/lustre/mlnvme/data/bshomali_hpc-data_workspace_main/logs/MTP_1B/%x-%j.out
#SBATCH --error=/lustre/mlnvme/data/bshomali_hpc-data_workspace_main/logs/MTP_1B/%x-%j.err
#SBATCH --gpus-per-node=4
#SBATCH --nodes=4
#SBATCH --partition=sgpu_long
#SBATCH --time=5-00:00:00
#SBATCH --account=ag_ifi_wrobel
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3


## --qos=normal
## --exclusive
## --mem=0       
export CXX=g++
export CC=gcc
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10

export DEVICES_PER_NODE=4
export NUM_NODES="$SLURM_JOB_NUM_NODES"
 
set -x

WORKING_DIR=/home/bshomali_hpc/modalities/
cd $WORKING_DIR


module load Miniforge3
source $(conda info --base)/etc/profile.d/conda.sh
# conda activate /leonardo_work/EUHPC_D21_101/bshomali/conda-envs/modaltities  
# conda init
conda activate modalities
source ~/export_env_variables.sh


CONFIG_FILES=(
  "/home/bshomali_hpc/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/1B_configs/3MTP_12_WS_woShifted_HiddenStateAligned_woAlign_universalGate_woSigmoid_scalerBias_HigherAlign.yaml"
  "/home/bshomali_hpc/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/1B_configs/5MTP_12_WS_woShifted_HiddenStateAligned_woAlign_universalGate_woSigmoid_scalerBias_HigherAlign.yaml"
  "/home/bshomali_hpc/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/1B_configs/7MTP_12_WS_woShifted_HiddenStateAligned_woAlign_universalGate_woSigmoid_scalerBias_HigherAlign.yaml"
  "/home/bshomali_hpc/modalities/config_files/training/nemtron_24_effective_layers_proj/various_k/math/partial_MTP/partial_MTP_gating/attention/different_blocksizes/1B_configs/9MTP_12_WS_woShifted_HiddenStateAligned_woAlign_universalGate_woSigmoid_scalerBias_HigherAlign.yaml"
)

export CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((6000 + SLURM_ARRAY_TASK_ID))
 

echo "START TIME: $(date)"
begin_time=$(date +%s) 

srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$DEVICES_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  $(which modalities) run --config_file_path $CONFIG_FILE

echo "END TIME: $(date)"
echo "=== FINISHED ==="
end_time=$(date +%s)

elapsed_seconds=$((end_time - begin_time))
elapsed_hours=$(awk "BEGIN {printf \"%.2f\", $elapsed_seconds/3600}")

echo "Elapsed time: $elapsed_hours hours"
 