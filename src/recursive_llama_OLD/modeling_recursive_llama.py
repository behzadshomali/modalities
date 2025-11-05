from transformers import LlamaForCausalLM
from .configuration_recursive_llama import RecursiveLlamaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class RecursiveLlamaForCausalLM(LlamaForCausalLM):
    config_class = RecursiveLlamaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self._recursion_applied = False
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Apply recursion AFTER loading pretrained weights
        if isinstance(model, cls) and hasattr(model, "config"):
            cfg = model.config
            if getattr(cfg, "recursive_start_layer", None) is not None:
                model._apply_block_recursion(
                    cfg.recursive_start_layer,
                    cfg.recursive_end_layer,
                    cfg.num_recursions,
                    cfg.sample_random_recursion,
                    cfg.neft,
                    cfg.neft_alpha
                )
                model._recursion_applied = True

        return model
    
    def _apply_block_recursion(self, start_layer, end_layer, num_recursions, sample_random_recursion, neft=False, neft_alpha=None):
        """Apply your recursion logic here"""
        n_layers = len(self.model.layers)
        
        # Extract the block
        layer_block = nn.ModuleList([self.model.layers[i] for i in range(start_layer, end_layer + 1)])
        
        class BlockRecursiveModule(nn.Module):
            def __init__(self, layer_block, num_recursions, sample_random_recursion, track_diagnostics):
                super().__init__()
                self.layer_block = layer_block
                self.num_recursions = num_recursions
                self.sample_random_recursion = sample_random_recursion
                self.track_diagnostics = track_diagnostics
                self.output_dir = './'
                self.neft = neft
                self.neft_alpha = neft_alpha

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

                        if self.neft and self.training:
                            alpha = self.neft_alpha
                            L = hidden_states.shape[-2]
                            d = hidden_states.shape[-1]
                            noise = torch.rand_like(hidden_states) * 2 -1 # range: [-1,1]
                            scaled_noise = noise * alpha / ((L*d)**0.5)
                            hidden_states = hidden_states + scaled_noise


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
        
        block_module = BlockRecursiveModule(layer_block, num_recursions, sample_random_recursion, track_diagnostics=True)
        
        # Build new ModuleList
        new_layers = []
        for i in range(start_layer):
            new_layers.append(self.model.layers[i])
        new_layers.append(block_module)
        for i in range(end_layer + 1, n_layers):
            new_layers.append(self.model.layers[i])
        
        self.model.layers = nn.ModuleList(new_layers)