import random

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RecursiveDecoderLayer(nn.Module):
    """Wraps a decoder layer with fixed recursion"""
    def __init__(self, original_layer, num_recursions=3):
        super().__init__()
        self.layer = original_layer
        self.num_recursions = num_recursions
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, 
                position_embeddings=None, **kwargs):
        
        # Apply the same layer multiple times
        outputs = None
        for i in range(self.num_recursions):
            outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs
            )
            hidden_states = outputs[0].unsqueeze(0)
        
        return hidden_states
    

def add_recursion_to_llama(model, layer_indices, num_recursions=3):
    """
    Add recursion to specific layers of a Llama model
    
    Args:
        model: Loaded Llama model
        layer_indices: List of layer indices to make recursive (e.g., [10, 15, 20])
        num_recursions: Number of times to repeat each layer (fixed mode)
    """
    for idx in layer_indices:
        if idx >= len(model.model.layers):
            print(f"Warning: Layer index {idx} out of range. Skipping.")
            continue
        
        original_layer = model.model.layers[idx]
        model.model.layers[idx] = RecursiveDecoderLayer(
            original_layer,
            num_recursions=num_recursions
        )
        print(f"Added fixed recursion to layer {idx} ({num_recursions} iterations)")
    
    return model


def add_block_recursion_to_llama(model, start_layer, end_layer, num_recursions=3, sample_random_recursion=False):
    """
    Replace layers[start_layer:end_layer+1] with a single module that applies
    that block num_recursions times. Correctly handles KV caching.
    """
    n_layers = len(model.model.layers)
    if start_layer < 0 or end_layer >= n_layers or start_layer > end_layer:
        raise ValueError(f"Invalid layer range: [{start_layer}, {end_layer}]")
    
    # Extract the block of layers
    layer_block = nn.ModuleList([model.model.layers[i] for i in range(start_layer, end_layer + 1)])

    # The wrapper module that *looks like one decoder layer* to the caller
    class BlockRecursiveModule(nn.Module):
        def __init__(self, layer_block, num_recursions):
            super().__init__()
            self.layer_block = layer_block
            self.num_recursions = num_recursions
            self.sample_random_recursion = sample_random_recursion
        
        def forward(
            self, 
            hidden_states, 
            attention_mask=None, 
            position_ids=None,
            past_key_value=None, 
            output_attentions=False, 
            use_cache=False,
            position_embeddings=None, 
            **kwargs
        ):
            
            outputs = None
            
            # This will hold the cache from the *previous iteration*
            # For iteration 0, it's the cache passed into the module.
            iter_past_key_values = past_key_value 

            final_present_key_values = () if use_cache else None

            if self.sample_random_recursion:
                num_recursions = random.randint(0, self.num_recursions)
            else: 
                num_recursions = self.num_recursions

            for iteration in range(num_recursions):
                
                # new_caches_this_iteration = () if use_cache else None
                
                for i, layer in enumerate(self.layer_block):
                    
                    # # FIX 2: Get the *specific* layer's cache for this iteration
                    # # 'iter_past_key_values' is a tuple of caches: (cache_8, cache_9, ...)
                    # layer_past_key_value = None
                    # if iter_past_key_values is not None:
                    #     try:
                    #         layer_past_key_value = iter_past_key_values[i]
                    #     except (IndexError, TypeError):
                    
                    #         layer_past_key_value = None
                            
                    outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None, 
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = outputs[0].unsqueeze(0)
                    
            
            return hidden_states


    # Create module and replace the slice in the original layers list
    block_module = BlockRecursiveModule(layer_block, num_recursions=num_recursions)

    # Build new ModuleList
    new_layers = []
    for i in range(start_layer):
        new_layers.append(model.model.layers[i])
    new_layers.append(block_module)
    for i in range(end_layer + 1, n_layers):
        new_layers.append(model.model.layers[i])

    # Reassign layers to a new ModuleList
    model.model.layers = nn.ModuleList(new_layers)

    print(f"Replaced layers [{start_layer}:{end_layer}] with a recursive block ({num_recursions} iterations).")
    return model