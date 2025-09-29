export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

BASE_DIR="/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_05-17_02_58_Teuken3.7B_IT_OpenMathInstruct-2/2025_09_06-12_40_07_lora+_rank8_alpha16_1M(Markus)/2025_09_06-12_43_03/"


# Loop through directories under BASE_DIR
for dir in "$BASE_DIR"/*/; do
    # Check if it's a directory
    [ -d "$dir" ] || continue

    # Look for checkpoint-* directories inside
    for ckpt in "$dir"/*; do
        if [[ "$ckpt" == *"checkpoint"* ]]; then
            if [ -d "$ckpt" ]; then
                CUDA_VISIBLE_DEVICES=2 accelerate launch \
                    --main_process_port 2000 \
                    --num_processes 1 \
                    --num_machines 1 \
                    -m \
                    lighteval accelerate \
                    "model_name=$ckpt,trust_remote_code=True,use_chat_template=True" \
                    "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
                    --use-chat-template;
            fi
        fi
    done
done






# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 2 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_04-17_30_34/checkpoint-2000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir "/raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_09_04-17_30_34/rank8/checkpoint-2000/" \
#     --use-chat-template;

# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 2 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_04-17_30_34/checkpoint-4000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir "/raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_09_04-17_30_34/rank8/checkpoint-4000/" \
#     --use-chat-template;

# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 2 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_04-17_30_34/checkpoint-6000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir "/raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_09_04-17_30_34/rank8/checkpoint-6000/" \
#     --use-chat-template;

# CUDA_VISIBLE_DEVICES=2 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 2 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_04-17_30_34/checkpoint-7194/lora_merged/,trust_remote_code=True,use_chat_template=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir "/raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_09_04-17_30_34/rank8/checkpoint-7194/" \
#     --use-chat-template;