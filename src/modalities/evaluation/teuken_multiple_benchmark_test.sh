export HF_HOME=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/
export TRANSFORMERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/transformers
export HF_DATASETS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/datasets
export HF_TOKENIZERS_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/tokenizers
export HF_HUB_CACHE=/raid/s3/opengptx/behzad_shomali/custom_hf_cache/hub_cache

huggingface-cli login

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --main_process_port 2000 \
    --num_processes 2 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_05-17_03_49_Teuken3.7B_IT_OpenMathInstruct-2/2025_09_06-12_39_29_lora+_rank16_alpha32_1M(Markus)/2025_09_06-12_43_26/checkpoint-30000/lora_merged/,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|8|1" \
    --output-dir "/raid/s3/opengptx/behzad_shomali/evaluation_results/teuken3.7B_IT_LoRA-OpenMathInstruct-2/2025_09_04-17_30_34/rank8/checkpoint-2000/" \
    --use-chat-template

