import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import clone_chat_template
from huggingface_hub import HfApi

# Base model (same one you used during training)
base_model = "Behzadshomali/Teuken3.7B"

# Your LoRA model (local folder or HF Hub repo)
lora_model = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.73T_IT_GSM8K_socratic/2025_08_23-10_57_44/checkpoint-4645"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)

model, tokenizer, _ = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

# Load LoRA on top of base
model = PeftModel.from_pretrained(model, lora_model)

# Merge LoRA adapters into the base model
model = model.merge_and_unload()


revision = "2025_08_23-10_57_44-checkpoint-4645"
model.push_to_hub(
    "Behzadshomali/Teuken3.7B_IT_LoRA",
    revision=revision,
    private=True,             # optional
    use_temp_dir=True,        # makes sure full files are synced
    commit_message="Add merged LoRA model with custom code",
    safe_serialization=True
)

tokenizer.push_to_hub("Behzadshomali/Teuken3.7B_IT_LoRA", revision=revision)

api = HfApi()
api.upload_file(
    path_or_fileobj="/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py",       
    repo_id="Behzadshomali/Teuken3.7B_IT_LoRA",  
    path_in_repo="modeling_gpt2.py",     
    repo_type="model",
    revision=revision
)