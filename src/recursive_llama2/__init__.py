from recursive_llama2.recursive_llama import RecursiveLlamaForCausalLM, RecursiveLlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM

# Register the classes
AutoConfig.register("recursive-llama", RecursiveLlamaConfig)
AutoModelForCausalLM.register(RecursiveLlamaConfig, RecursiveLlamaForCausalLM)

print("✓ Recursive Llama classes registered!")