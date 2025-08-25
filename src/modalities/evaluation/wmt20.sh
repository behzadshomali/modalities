CUDA_VISIBLE_DEVICES=3 lighteval accelerate \
    "model_name=Behzadshomali/Multilingual5,revision=29.44B,trust_remote_code=True" \
    "lighteval|wmt20:de-en|0|1,lighteval|wmt20:en-de|0|1,lighteval|wmt20:fr-de|0|1,lighteval|wmt20:de-fr|0|1,lighteval|wmt20:de-en|10|1,lighteval|wmt20:en-de|10|1,lighteval|wmt20:fr-de|10|1,lighteval|wmt20:de-fr|10|1,lighteval|wmt20:de-en|3|1,lighteval|wmt20:en-de|3|1,lighteval|wmt20:fr-de|3|1,lighteval|wmt20:de-fr|3|1" \
    --output-dir /raid/s3/opengptx/behzad_shomali/wmt20_results/29-44B/ \
    --push-to-hub \
    --save-details \
    --results-org Behzadshomali
