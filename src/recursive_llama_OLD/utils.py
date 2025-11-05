import random
import pickle
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import random
from collections import defaultdict

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


def add_block_recursion_to_llama(model, start_layer, end_layer, num_recursions=3, sample_random_recursion=False, track_diagnostics=False, output_dir="."):
    """
    Replace layers[start_layer:end_layer+1] with a single module that applies
    that block num_recursions times. Correctly handles KV caching.
    
    Args:
        track_diagnostics: If True, enables tracking of activations and gradients
    """
    n_layers = len(model.model.layers)
    if start_layer < 0 or end_layer >= n_layers or start_layer > end_layer:
        raise ValueError(f"Invalid layer range: [{start_layer}, {end_layer}]")
    
    # Extract the block of layers
    layer_block = nn.ModuleList([model.model.layers[i] for i in range(start_layer, end_layer + 1)])

    class BlockRecursiveModule(nn.Module):
        def __init__(self, layer_block, num_recursions, sample_random_recursion, track_diagnostics):
            super().__init__()
            self.layer_block = layer_block
            self.num_recursions = num_recursions
            self.sample_random_recursion = sample_random_recursion
            self.track_diagnostics = track_diagnostics
            self.output_dir = output_dir

            if track_diagnostics:
                self.gradient_history = []    # Store gradients
                self.cosine_similarities = []  # Store cosine similarities between iterations
                self.step_count = 0


        
        def compute_cosine_similarity(self, tensor1, tensor2):
            """Compute token-wise cosine similarity between two tensors."""
            # Flatten to [batch_size * seq_len, hidden_dim]
            t1 = tensor1.reshape(-1, tensor1.shape[-1])
            t2 = tensor2.reshape(-1, tensor2.shape[-1])
            
            # Compute cosine similarity for each token
            cos_sim = F.cosine_similarity(t1, t2, dim=-1)
            return cos_sim
        
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    position_embeddings=None, **kwargs):
            
            outputs = None
            if self.sample_random_recursion:
                num_recursions = random.randint(0, self.num_recursions)
            else: 
                num_recursions = self.num_recursions

            # Clear tracking for this forward pass
            if self.track_diagnostics and self.training:
                prev_hidden_states = hidden_states.detach().clone()
            
            for iteration in range(num_recursions):                
                for layer_idx, (layer_name, layer) in enumerate(self.layer_block._modules.items()):
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

                    if self.track_diagnostics and hidden_states.requires_grad and self.training:
                        def make_grad_hook(layer_idx, iteration_idx):
                            def grad_hook(grad):
                                # record grad info
                                self.gradient_history.append({
                                    'layer_name': layer_name, # negative layer represents rec layers
                                    'is_recurrent': True,
                                    'iteration': iteration_idx,
                                    'step': self.step_count,
                                    'grad_norm': grad.norm().item(),
                                    'grad_mean': grad.mean().item(),
                                    'grad_std': grad.std().item(),
                                    'grad_max': grad.abs().max().item()
                                })
                                # increment step counter
                                self.step_count += 1
                                return grad
                            return grad_hook

                        # register hook with the current iteration/layer
                        hidden_states.register_hook(make_grad_hook(layer_idx, iteration))
                
                    elif not hidden_states.requires_grad and self.training:
                        print(layer_idx, layer)
            
                
                # Track activations and compute similarities after each iteration
                if self.track_diagnostics and self.training:
                    current_activation = hidden_states.detach().clone()
                    
                    # Compute cosine similarity with previous iteration
                    cos_sim = self.compute_cosine_similarity(prev_hidden_states, current_activation)
                    self.cosine_similarities.append({
                        'iteration': iteration,
                        'mean': cos_sim.mean().item(),
                        'min': cos_sim.min().item(),
                        'max': cos_sim.max().item(),
                        'std': cos_sim.std().item(),
                        'per_token': cos_sim.cpu().numpy()
                    })
                    
                    prev_hidden_states = current_activation
            
            return hidden_states
        
        def get_diagnostics(self):
            """Return diagnostic information about activations and gradients."""
            return {
                'cosine_similarities': self.cosine_similarities,
                'gradient_history': self.gradient_history,
            }
        
        def print_diagnostics(self):
            """Print a summary of diagnostic information."""
            
            if self.cosine_similarities:
                print("\nActivation Cosine Similarities (iteration-to-iteration):")
                print("-" * 70)
                for sim in self.cosine_similarities:
                    print(f"  Iteration {sim['iteration']}:")
                    print(f"    Mean: {sim['mean']:.6f} | Min: {sim['min']:.6f} | "
                          f"Max: {sim['max']:.6f} | Std: {sim['std']:.6f}")
                    
                    if sim['mean'] > 0.99:
                        print(f"    ⚠️  WARNING: Very high similarity ({sim['mean']:.6f}) - "
                              f"recursion may be inert!")
                print()
            else:
                print("\nNo activation similarities tracked yet.")
            
            if self.gradient_history:
                print("\nGradient Statistics (last 5 steps):")
                print("-" * 70)
                for grad_info in self.gradient_history[-5:]:
                    print(f"  Step {grad_info['step']}:")
                    print(f"    Norm: {grad_info['grad_norm']:.6f} | "
                          f"Mean: {grad_info['grad_mean']:.6f} | "
                          f"Std: {grad_info['grad_std']:.6f} | "
                          f"Max: {grad_info['grad_max']:.6f}")
                print()
            else:
                print("\nNo gradients tracked yet.")
            
            print("="*70 + "\n")


    # Create module and replace the slice in the original layers list
    block_module = BlockRecursiveModule(layer_block, num_recursions, sample_random_recursion, track_diagnostics)


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
    if track_diagnostics:
        register_global_gradient_tracking(model, block_module)
        print(f"Diagnostic tracking ENABLED")
    
    return model, block_module if track_diagnostics else None



def register_global_gradient_tracking(model, tracker):
    """
    Register hooks on all parameters in the model to track their gradients.
    
    Args:
        model: The full nn.Module (e.g., the Llama model)
        tracker: An object (e.g., block_module) with a `.gradient_history` list
                 and `.step_count` counter to store gradient statistics.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        def make_param_grad_hook(param_name):
            def hook(grad):
                if "model.layers." in param_name:
                    layer_idx = int(param_name.split(".")[2] )
                else:
                    layer_idx = -100
                tracker.gradient_history.append({
                    'layer_name': param_name,
                    'layer': layer_idx,
                    'step': tracker.step_count,
                    'grad_norm': grad.norm().item(),
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item(),
                    'grad_max': grad.abs().max().item(),
                })
                tracker.step_count += 1
                return grad
            return hook

        param.register_hook(make_param_grad_hook(name))
