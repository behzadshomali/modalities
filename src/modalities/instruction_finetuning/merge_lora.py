import os
import gc
import torch
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer
# from recursive_llama.utils import add_block_recursion_to_llama, add_recursion_to_llama

from peft import PeftModel
from trl import setup_chat_format
from huggingface_hub import HfApi

def get_trainable_weights(checkpoint_path, trainable_params):
    weights_all = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))

    weights_trainable = {}
    weights_lora = {}
    for k in weights_all:
        if "lora" in k:
            k_new = k.replace("default.", "") if "default." in k else k
            weights_lora[k_new] = weights_all[k]
        else:
            if any([n in k for n in trainable_params]):
                # len("base_model.model.") = 17
                weights_trainable[k[17:]] = weights_all[k]

    adapter_model = os.path.join(checkpoint_path, "adapter_model.bin")
    trainable_params = os.path.join(checkpoint_path, "trainable_params.bin")
    if not os.path.isfile(adapter_model):
        torch.save(weights_lora, adapter_model)
    torch.save(weights_trainable, trainable_params)


def merge_lora_adapter(
    lora_model, 
    base_model, 
    save_on_disk=True, 
    upload_to_HF=False,
    **kwargs_HF
):
    save_path = None
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )

    if "llama" in base_model:
        print("X"*10, "Llama model has been modified!", "X"*10)
        recursion_config = kwargs_HF["recursion_settings"]
        raise ValueError("Not imeplemented yet!")
        # if recursion_config["type"] == "block":
        #     model = add_block_recursion_to_llama(
        #         model,
        #         start_layer=recursion_config["start_layer"],     
        #         end_layer=recursion_config["end_layer"],     
        #         num_recursions=recursion_config["num_recursions"]
        #     )
        # elif recursion_config["type"] == "layer":
        #     model = add_recursion_to_llama(
        #         model,
        #         layer_indices=recursion_config["layer_indices"],
        #         num_recursions=recursion_config["num_recursions"]
        #     )
        # else:
            # raise ValueError("The recursion type must be from [layer/block]")



    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # get_trainable_weights(lora_model, ['embed', 'norm'])

    trainable_params = os.path.join(lora_model, "trainable_params.bin")
    # if os.path.isfile(trainable_params):
    print("Loading trainable parameters from:", trainable_params)
    model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
    
    model = PeftModel.from_pretrained(
        model,
        lora_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
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

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return save_path

def apply_merge_on_all_checkpints(base_model, experiment_dir):
    for directory in os.listdir(experiment_dir):
        if directory.startswith("checkpoint"):
            checkpoint_dir = os.path.join(experiment_dir, directory)
            if any(["lora_merged" in sub_dir for sub_dir in os.listdir(checkpoint_dir)]):
                continue
            print("Merging:", checkpoint_dir)
            merge_lora_adapter(
                lora_model=checkpoint_dir, 
                base_model=base_model,
                save_on_disk=True,  
                upload_to_HF=False
            )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    base_model =  "Behzadshomali/Teuken3.7B"
    # Qwen/Qwen3-4B-Base
    # base_model = "meta-llama/Llama-3.2-3B"
    # experiment_dir = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/Teuken3.7B_IT_OpenMathInstruct-2/2025_09_05-17_03_49_Teuken3.7B_IT_OpenMathInstruct-2/2025_09_06-12_39_29_lora+_rank16_alpha32_1M(Markus)/2025_09_06-12_43_26_lora+_rank16_alpha32_1M(Markus)/2025_09_09-12_18_24_lora+_rank16_alpha32_1M(Markus)/2025_09_09-12_18_50_lora+_rank16_alpha32_1M(Markus)/2025_09_09-12_19_31_lora+_rank16_alpha32_1M(Markus)/2025_09_09-12_20_30_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_11_10_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_12_29_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_13_39_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_15_40_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_22_31_lora+_rank16_alpha32_1M(Markus)/2025_09_09-13_26_29_lora+_rank16_alpha32_1M(Markus)/2025_09_09-17_35_41_lora+_rank16_alpha32_1M(Markus)/2025_09_09-17_36_54"
    experiment_dir = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/_lora+_rank16_alpha32_1M(Markus)/2025_09_10-20_00_40/"
    # apply_merge_on_all_checkpints(base_model, experiment_dir)
    merge_lora_adapter(
        lora_model="/raid/s3/opengptx/behzad_shomali/instruction_tuning/_lora+_rank16_alpha32_1M_lmHead/2025_09_15-12_33_51/checkpoint-16000/",
        base_model=base_model,
        save_on_disk=True,  
        upload_to_HF=False
    )