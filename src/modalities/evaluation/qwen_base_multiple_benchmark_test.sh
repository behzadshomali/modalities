export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --main_process_port 2000 \
    --num_processes 1 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.73T_IT_OpenMathInstruct-2/2025_08_30-19_33_56/rank8/checkpoint-9084/lora_merged,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|5|1,leaderboard|gsm8k|0|0,leaderboard|hellaswag|0|0,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1,leaderboard|mmlu:high_school_mathematics|0|0,leaderboard|mmlu:high_school_mathematics|5|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/evaluation_results/Qwen/2025_08_30-19_33_56/rank8/checkpoint-9084/ \
    --use-chat-template