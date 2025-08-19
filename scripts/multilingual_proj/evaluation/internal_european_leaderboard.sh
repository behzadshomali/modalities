CUDA_VISIBLE_DEVICES=3 lighteval accelerate \
    "model_name=Behzadshomali/Multilingual5,revision=29.44B,trust_remote_code=True" \
    "lighteval|arc:easy|3|1,leaderboard|hellaswag|3|1,helm|mmlu|3|1,leaderboard|gsm8k|3|1,leaderboard|truthfulqa:mc|3|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/internal_european_leaderboard/29-44B/ \
    --push-to-hub \
    --save-details \
    --results-org Behzadshomali
