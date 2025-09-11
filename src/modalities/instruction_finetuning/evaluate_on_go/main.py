import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from evaluation_utils import setup_wandb, start_evaluation_loop
from utils import load_config

if __name__ == "__main__":
    experiment_dir = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_02-17_50_17/"
    config_path = "/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/configs/OpenMathInstruct-2+norm_embed_rank16.yaml"
    additional_checkpoint_to_evaluate = [
        "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_02-17_50_17/checkpoint-2000/",
        "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_02-17_50_17/checkpoint-4000/",
        "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_02-17_50_17/checkpoint-6000/"
    ]


    config = load_config(config_path)
    setup_wandb(config)
    start_evaluation_loop(config, experiment_dir, additional_checkpoint_to_evaluate)