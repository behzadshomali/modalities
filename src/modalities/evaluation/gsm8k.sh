CUDA_VISIBLE_DEVICES=1,2,4 accelerate launch \
    --main_process_port 2000 \
    --num_processes 3 \
    --num_machines 1 \
    -m \
    lighteval accelerate \
    "model_name=Behzadshomali/Teuken3.7B_IT_LoRA,revision=2025_08_23-10_57_44-checkpoint-4645,trust_remote_code=True,use_chat_template=True" \
    "leaderboard|gsm8k|8|1,leaderboard|gsm8k|5|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/evaluation_results/gsm8k/teuken3.7B_IT/2025_08_23-10_57_44/checkpoint-4645/
