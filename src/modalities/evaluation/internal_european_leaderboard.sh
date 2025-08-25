CUDA_VISIBLE_DEVICES=3 lighteval accelerate \
    "model_name=Behzadshomali/Multilingual5,revision=29.44B,trust_remote_code=True" \
    "lighteval|arc:easy|0|0,leaderboard|hellaswag|0|0,helm|mmlu|0|0,leaderboard|gsm8k|0|0" \
    --output-dir /raid/s3/opengptx/behzad_shomali/internal_european_leaderboard/29-44B/ \
    --push-to-hub \
    --save-details \
    --results-org Behzadshomali
