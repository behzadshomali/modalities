import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from recursive_llama import RecursiveLlamaConfig, RecursiveLlamaForCausalLM # Import our new classes

# --- Define Your Conversion Parameters ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
NEW_MODEL_PATH = "./my-recursive-llama-7b" # Where to save the new model
RECURSION_START = 8
RECURSION_END = 15
NUM_RECURSIONS = 3

print("Loading base model...")
base_model = LlamaForCausalLM.from_pretrained(BASE_MODEL_ID)
print("Base model loaded.")

# --- 1. Create the new config ---
print("Creating new recursive config...")
config = RecursiveLlamaConfig.from_pretrained(
    BASE_MODEL_ID,
    original_num_hidden_layers=base_model.config.num_hidden_layers,
    recursion_start_layer=RECURSION_START,
    recursion_end_layer=RECURSION_END,
    num_recursions=NUM_RECURSIONS,
    sample_random_recursion=True, # Set your defaults
    track_diagnostics=True,
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

# --- 4. Save the new, loadable model ---
print(f"Saving new model to {NEW_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(NEW_MODEL_PATH)
model.save_pretrained(NEW_MODEL_PATH)

print("Conversion complete!")