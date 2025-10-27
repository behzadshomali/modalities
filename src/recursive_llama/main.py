import os
from recursive_llama.utils import add_block_recursion_to_llama
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

TOKEN = os.environ["HF_TOKEN"]

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token=TOKEN).to("cuda")

question = "Once upon a time there was"
inputs = tokenizer(question, return_tensors="pt").to("cuda")

new_model = add_block_recursion_to_llama(
    deepcopy(model), 
    start_layer=8,     
    end_layer=11,       # 4-layer block
    num_recursions=0
)

outputs = new_model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    top_p=0.9,
    # cache_position=None,
    use_cache=False,
    temperature=0.2,
)
print(tokenizer.decode(outputs.detach().cpu().numpy()[0]))