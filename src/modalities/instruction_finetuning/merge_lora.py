import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import setup_chat_format
from huggingface_hub import HfApi



def merge_lora_adapter(
        lora_model, 
        base_model, 
        save_on_disk=True, 
        upload_to_HF=False,
        **kwargs_HF
    ):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load LoRA on top of base
    model = PeftModel.from_pretrained(model, lora_model)

    # Merge LoRA adapters into the base model
    model = model.merge_and_unload()

    if save_on_disk:
        save_path = os.path.join(lora_model, "lora_merged")
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

        # the modeling_gpt2.py file is missing after merge,
        # therefore it is copied manually
        custom_modeling_src = "/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py"
        custom_modeling_dst = os.path.join(save_path, "modeling_gpt2.py")
        shutil.copy(custom_modeling_src, custom_modeling_dst)

    if upload_to_HF:
        revision = kwargs_HF["revision"]
        repo_id = kwargs_HF["repo_id"]
        commit_message = kwargs_HF.get("commit_message", "Add merged LoRA model with custom code")

        model.push_to_hub(
            repo_id,
            revision=revision,
            private=True,
            use_temp_dir=True, # makes sure full files are synced
            commit_message=commit_message,
            safe_serialization=True
        )
        tokenizer.push_to_hub(repo_id, revision=revision)

        api = HfApi()
        api.upload_file(
            path_or_fileobj="/home/behzad_shomali/modalities/src/modalities/instruction_finetuning/modeling_gpt2.py",       
            repo_id=repo_id,  
            path_in_repo="modeling_gpt2.py",     
            repo_type="model",
            revision=revision
        )

if __name__ == "__main__":
    base_model =  "Qwen/Qwen3-4B-Base" #"Behzadshomali/Teuken3.7B"
    lora_model = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.73T_IT_OpenMathInstruct-2/2025_08_30-19_33_56/rank8/checkpoint-9084/"
    merge_lora_adapter(
        lora_model, 
        base_model,
        save_on_disk=True,
        upload_to_HF=False
    )