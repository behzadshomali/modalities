export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
    --main_process_port 2000 \
    --num_processes 2 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Llama3.1_3B_IT_OpenMathInstruct-2/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_01-18_41_56/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_01-20_53_57/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_02-11_13_11/checkpoint-2000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/instruction_tuning/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_02-11_13_11/checkpoint-2000/rank16/ \
    --use-chat-template;

CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
    --main_process_port 2000 \
    --num_processes 2 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Llama3.1_3B_IT_OpenMathInstruct-2/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_01-18_41_56/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_01-20_53_57/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_02-11_13_11/checkpoint-4338/lora_merged/,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/instruction_tuning/Llama3.1_3B_IT_OpenMathInstruct-2/2025_09_02-11_13_11/checkpoint-4338/rank16/ \
    --use-chat-template;

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 2 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=meta-llama/Llama-3.2-3B,trust_remote_code=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir /raid/s3/opengptx/behzad_shomali/instruction_tuning/meta-llama/Llama-3.2-3B-base/