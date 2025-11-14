import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from recursive_llama import RecursiveLlamaConfig, RecursiveLlamaForCausalLM # Import our new classes

# --- Define Your Conversion Parameters ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
NEW_MODEL_PATH = "./my-recursive-llama-7b" # Where to save the new model
RECURSION_START = 0
RECURSION_END = 0
NUM_RECURSIONS = 1

print("Loading base model...")
base_model = LlamaForCausalLM.from_pretrained(BASE_MODEL_ID)
print("Base model loaded.")

# --- 1. Create the new config ---
print("Creating new recursive config...")
config = RecursiveLlamaConfig(
    model_name=BASE_MODEL_ID,
    original_num_hidden_layers=base_model.config.num_hidden_layers,
    recursion_start_layer=RECURSION_START,
    recursion_end_layer=RECURSION_END,
    num_recursions=NUM_RECURSIONS,
    sample_random_recursion=False, # Set your defaults
    track_diagnostics=False,
    
    
)

# --- 2. Create the new model (with random weights) ---
print("Initializing new recursive architecture...")
# Because we registered our class, LlamaForCausalLM(config) would also work
model = RecursiveLlamaForCausalLM(config) 

# --- 3. Copy weights from base_model to model ---
print("Copying weights...")

# Embeddings and final normalization
model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
model.model.norm.load_state_dict(base_model.model.norm.state_dict())
model.lm_head.load_state_dict(base_model.lm_head.state_dict())

# Layers BEFORE the block
model.model.layers[:RECURSION_START].load_state_dict(
    base_model.model.layers[:RECURSION_START].state_dict()
)

# Layers INTO the block
# model.model.layers[RECURSION_START] is our BlockRecursiveModule
model.model.layers[RECURSION_START].layer_block.load_state_dict(
    base_model.model.layers[RECURSION_START : RECURSION_END + 1].state_dict()
)

# Layers AFTER the block
# The new index is RECURSION_START + 1
# The original index is RECURSION_END + 1
model.model.layers[RECURSION_START + 1 :].load_state_dict(
    base_model.model.layers[RECURSION_END + 1 :].state_dict()
)

print("Weight copy complete.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
model = model.to("cuda")
base_model = base_model.to("cuda")


MAX_NEW_TOKENS = 200

# model.model.layers[RECURSION_START]

with torch.no_grad():
    model.model.layers[RECURSION_START].layer_block[0].mlp.down_proj.weight.data *= 5

prompt = "The key to life is to"
input_ = tokenizer(prompt, return_tensors="pt").to("cuda")
output_w_cache = model.generate(**input_, use_cache=True, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
print("WO CACHE ....")
output_wo_cache = model.generate(**input_, use_cache=False, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
print("IS EQUAL?", torch.allclose(output_w_cache, output_wo_cache))



base_model = LlamaForCausalLM.from_pretrained(BASE_MODEL_ID).to("cuda")

with torch.no_grad():
    base_model.model.layers[RECURSION_START].mlp.down_proj.weight.data *= 5


output_base_w_cache = base_model.generate(**input_, do_sample=False, use_cache=True, max_new_tokens=MAX_NEW_TOKENS)
print("BASE MODEL ...")
output_base_wo_cache = base_model.generate(**input_, do_sample=False, use_cache=False, max_new_tokens=MAX_NEW_TOKENS)
print("IS EQUAL TO BASE?", torch.allclose(output_base_w_cache, output_base_wo_cache))
print("IS EQUAL TO BASE W CACHE?", torch.allclose(output_w_cache, output_base_w_cache))
print("IS EQUAL TO BASE WO CACHE?", torch.allclose(output_wo_cache, output_base_wo_cache))



print(tokenizer.decode(output_w_cache[0]))
print(tokenizer.decode(output_wo_cache[0]))
print(tokenizer.decode(output_base_w_cache[0]))


if True:
    pass
