import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from recursive_llama import RecursiveLlamaConfig, RecursiveLlamaForCausalLM # Import our new classes

# --- Define Your Conversion Parameters ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
NEW_MODEL_PATH = "./my-recursive-llama-7b" # Where to save the new model
RECURSION_START = 1
RECURSION_END = 15
NUM_RECURSIONS = 10

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




# finetuned_model_name = "/raid/s3/opengptx/behzad_shomali/instruction_tuning/pretrained_llama/dummy/2025_11_02-20_23_15/checkpoint-100"
# model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, output_hidden_states=True, trust_remote_code=True).to("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)

# model = model.eval()
# model.model.layers[8].num_recursions = 2
# prompt = "Yesterday I saw"
prompt = "The key to life is fsd "
input_ = tokenizer(prompt, return_tensors="pt").to("cuda")


output_w_cache = model(**input_, use_cache=True, do_sample=False)["logits"]

print("WO CACHE ....")

output_wo_cache = model(**input_, use_cache=False, do_sample=False)["logits"]

print("IS EQUAL?", torch.allclose(output_w_cache, output_wo_cache))
if True:
    pass
