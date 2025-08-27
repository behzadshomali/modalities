export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

# CUDA_VISIBLE_DEVICES=6 accelerate launch \
#     --main_process_port 2000 \
#     --num_processes 1 \
#     --num_machines 1 \
#     -m \
#     lighteval accelerate \
#     "model_name=Behzadshomali/Teuken3.7B_IT_LoRA-OpenMathInstruct-2,revision=2025_08_26-09_59_53-checkpoint-5274,trust_remote_code=True,use_chat_template=True" \
#     "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|0|0,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|0|0,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
#     --output-dir /raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_08_26-09_59_53/checkpoint-5274/ \
#     --use-chat-template

CUDA_VISIBLE_DEVICES=6 accelerate launch \
    --main_process_port 2000 \
    --num_processes 1 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=Behzadshomali/Teuken3.7B,trust_remote_code=True" \
    "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|0|0,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|0|0,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B/ \
