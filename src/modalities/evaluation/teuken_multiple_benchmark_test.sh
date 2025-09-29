export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

CUDA_VISIBLE_DEVICES=2 accelerate launch \
    --main_process_port 2000 \
    --num_processes 1 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/_lora+_rank16_alpha32_1M_lmHead/2025_09_15-12_33_51/checkpoint-16000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|5|1,leaderboard|hellaswag|5|1,leaderboard|truthfulqa:mc|5|1,leaderboard|arc:challenge|5|1" \
    --use-chat-template