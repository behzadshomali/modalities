import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import clone_chat_template, setup_chat_format
from huggingface_hub import HfApi
from utils import print_trainable_params

# Base model (same one you used during training)
base_model = "Behzadshomali/Teuken3.7B"

# Your LoRA model (local folder or HF Hub repo)
lora_model = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.73T_IT_OpenMathInstruct-2/2025_08_26-09_59_53_4096/checkpoint-15823/"
output_dir = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.73T_IT_OpenMathInstruct-2/2025_08_26-09_59_53_4096/checkpoint-15823/hf_converted/"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)

# model, tokenizer, _ = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
model, tokenizer = setup_chat_format(model, tokenizer)

# Load LoRA on top of base
model = PeftModel.from_pretrained(model, lora_model)
print_trainable_params(model)

# Merge LoRA adapters into the base model
model = model.merge_and_unload()

save_path = os.path.join(output_dir)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

# Optionally copy your custom modeling file
custom_modeling_src = "/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py"
custom_modeling_dst = os.path.join(save_path, "modeling_gpt2.py")
shutil.copy(custom_modeling_src, custom_modeling_dst)


# revision = "2025_08_26-09_59_53-checkpoint-5274"
# model.push_to_hub(
#     "Behzadshomali/Teuken3.7B_IT_LoRA-OpenMathInstruct-2",
#     revision=revision,
#     private=True,             # optional
#     use_temp_dir=True,        # makes sure full files are synced
#     commit_message="Add merged LoRA model with custom code",
#     safe_serialization=True
# )

# tokenizer.push_to_hub("Behzadshomali/Teuken3.7B_IT_LoRA-OpenMathInstruct-2", revision=revision)

# api = HfApi()
# api.upload_file(
#     path_or_fileobj="/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py",       
#     repo_id="Behzadshomali/Teuken3.7B_IT_LoRA-OpenMathInstruct-2",  
#     path_in_repo="modeling_gpt2.py",     
#     repo_type="model",
#     revision=revision
# )