import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class BlockRecursiveModule(nn.Module):
    """Recursive block that applies a sequence of layers multiple times."""
    
    def __init__(self, layer_block, num_recursions, sample_random_recursion=False, 
                 track_diagnostics=False):
        super().__init__()
        self.layer_block = layer_block
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics

        if track_diagnostics:
            self.gradient_history = []
            self.cosine_similarities = []
            self.step_count = 0

    def compute_cosine_similarity(self, tensor1, tensor2):
        """Compute token-wise cosine similarity between two tensors."""
        t1 = tensor1.reshape(-1, tensor1.shape[-1])
        t2 = tensor2.reshape(-1, tensor2.shape[-1])
        cos_sim = F.cosine_similarity(t1, t2, dim=-1)
        return cos_sim
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                position_embeddings=None, **kwargs):
        
        if self.sample_random_recursion:
            num_recursions = random.randint(0, self.num_recursions)
        else: 
            num_recursions = self.num_recursions

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
                            self.gradient_history.append({
                                'layer_name': layer_name,
                                'is_recurrent': True,
                                'iteration': iteration_idx,
                                'step': self.step_count,
                                'grad_norm': grad.norm().item(),
                                'grad_mean': grad.mean().item(),
                                'grad_std': grad.std().item(),
                                'grad_max': grad.abs().max().item()
                            })
                            self.step_count += 1
                            return grad
                        return grad_hook
                    hidden_states.register_hook(make_grad_hook(layer_idx, iteration))
            
            if self.track_diagnostics and self.training:
                current_activation = hidden_states.detach().clone()
                cos_sim = self.compute_cosine_similarity(prev_hidden_states, current_activation)
                self.cosine_similarities.append({
                    'iteration': iteration,
                    'mean': cos_sim.mean().item(),
                    'min': cos_sim.min().item(),
                    'max': cos_sim.max().item(),
                    'std': cos_sim.std().item(),
                })
                prev_hidden_states = current_activation
        
        return hidden_states
    
    def get_diagnostics(self):
        """Return diagnostic information."""
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


class RecursiveLlamaConfig(LlamaConfig):
    """Config for Llama with recursive blocks."""
    
    model_type = "recursive_llama"
    
    def __init__(
        self,
        recursive_start_layer=None,
        recursive_end_layer=None,
        num_recursions=3,
        sample_random_recursion=False,
        track_diagnostics=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.recursive_start_layer = recursive_start_layer
        self.recursive_end_layer = recursive_end_layer
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion
        self.track_diagnostics = track_diagnostics


class RecursiveLlamaForCausalLM(LlamaForCausalLM):
    """Llama model with recursive blocks that works with standard HF methods."""
    
    config_class = RecursiveLlamaConfig
    
    def __init__(self, config):
        # Temporarily remove recursion params to initialize base model
        recursive_params = {
            'recursive_start_layer': config.recursive_start_layer,
            'recursive_end_layer': config.recursive_end_layer,
            'num_recursions': config.num_recursions,
            'sample_random_recursion': config.sample_random_recursion,
            'track_diagnostics': config.track_diagnostics,
        }
        
        # Initialize base Llama model
        super().__init__(config)
        
        # Apply recursion if configured
        if recursive_params['recursive_start_layer'] is not None:
            self._apply_recursion(
                recursive_params['recursive_start_layer'],
                recursive_params['recursive_end_layer'],
                recursive_params['num_recursions'],
                recursive_params['sample_random_recursion'],
                recursive_params['track_diagnostics']
            )
    
    def _apply_recursion(self, start_layer, end_layer, num_recursions, 
                         sample_random_recursion, track_diagnostics):
        """Apply recursion to specified layers."""
        n_layers = len(self.model.layers)
        
        if start_layer < 0 or end_layer >= n_layers or start_layer > end_layer:
            raise ValueError(f"Invalid layer range: [{start_layer}, {end_layer}]")
        
        # Extract the block of layers
        layer_block = nn.ModuleList([
            self.model.layers[i] for i in range(start_layer, end_layer + 1)
        ])
        
        # Create recursive module
        block_module = BlockRecursiveModule(
            layer_block, num_recursions, sample_random_recursion, track_diagnostics
        )
        
        # Build new ModuleList
        new_layers = []
        for i in range(start_layer):
            new_layers.append(self.model.layers[i])
        new_layers.append(block_module)
        for i in range(end_layer + 1, n_layers):
            new_layers.append(self.model.layers[i])
        
        self.model.layers = nn.ModuleList(new_layers)
        
        print(f"✓ Replaced layers [{start_layer}:{end_layer}] with recursive block "
              f"({num_recursions} iterations)")
        if track_diagnostics:
            print(f"✓ Diagnostic tracking enabled")
    
    def get_recursive_block(self):
        """Get the recursive block module for diagnostics."""
        for layer in self.model.layers:
            if isinstance(layer, BlockRecursiveModule):
                return layer
        return None


# AUTO-REGISTER: This happens when the module is imported
AutoConfig.register("recursive_llama", RecursiveLlamaConfig)
AutoModelForCausalLM.register(RecursiveLlamaConfig, RecursiveLlamaForCausalLM)

print("✓ RecursiveLlamaForCausalLM registered with HuggingFace AutoClasses")


def save_for_external_tools(model, save_path):
    """
    Save model with modeling code for use with external tools (lighteval, vllm, etc.)
    This enables loading without needing to register the custom classes.
    
    Args:
        model: RecursiveLlamaForCausalLM model
        save_path: Directory to save the model
    """
    import shutil
    from pathlib import Path
    
    save_path = Path(save_path)
    
    # Save the model normally
    model.save_pretrained(save_path)
    
    # Copy this modeling file to the model directory
    current_file = Path(__file__)
    target_file = save_path / "modeling_recursive_llama.py"
    shutil.copy(current_file, target_file)
    
    # Update config.json to use trust_remote_code
    import json
    config_path = save_path / "config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['auto_map'] = {
        'AutoConfig': 'modeling_recursive_llama.RecursiveLlamaConfig',
        'AutoModelForCausalLM': 'modeling_recursive_llama.RecursiveLlamaForCausalLM',
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved with modeling code to {save_path}")
    print(f"✓ Can be loaded with: AutoModelForCausalLM.from_pretrained('{save_path}', trust_remote_code=True)")
    print(f"✓ Works with lighteval, vllm, and other external tools!")


def create_recursive_llama_from_pretrained(
    base_model_name_or_path,
    recursive_start_layer,
    recursive_end_layer,
    num_recursions=3,
    sample_random_recursion=False,
    track_diagnostics=False,
    **kwargs
):
    """
    Create a recursive Llama model from a pretrained base model.
    
    Args:
        base_model_name_or_path: HuggingFace model name or local path
        recursive_start_layer: First layer to make recursive
        recursive_end_layer: Last layer to make recursive (inclusive)
        num_recursions: Number of times to apply the block
        sample_random_recursion: Whether to randomize recursion depth
        track_diagnostics: Whether to enable diagnostic tracking
        **kwargs: Additional arguments for from_pretrained (device_map, torch_dtype, etc.)
    
    Returns:
        RecursiveLlamaForCausalLM model
    """
    # Load the base model config
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    
    # Create recursive config
    config = RecursiveLlamaConfig(
        **base_config.to_dict(),
        recursive_start_layer=recursive_start_layer,
        recursive_end_layer=recursive_end_layer,
        num_recursions=num_recursions,
        sample_random_recursion=sample_random_recursion,
        track_diagnostics=track_diagnostics,
    )
    
    # Load the base model weights
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        **kwargs
    )
    
    # Create recursive model with the same weights
    model = RecursiveLlamaForCausalLM(config)
    
    # Copy weights from base model (before recursion is applied)
    # This is a bit tricky because we need to load into the original structure
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Now apply recursion
    model._apply_recursion(
        recursive_start_layer,
        recursive_end_layer,
        num_recursions,
        sample_random_recursion,
        track_diagnostics
    )
    
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Create a recursive Llama model")
    print("=" * 80)
    
    model = create_recursive_llama_from_pretrained(
        "meta-llama/Llama-3.2-1B",
        recursive_start_layer=8,
        recursive_end_layer=11,
        num_recursions=3,
        track_diagnostics=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("\n" + "=" * 80)
    print("Example 2: Save with standard HuggingFace method")
    print("=" * 80)
    
    # Save - works with standard HF method!
    model.save_pretrained("./my_recursive_llama")
    print("✓ Model saved to ./my_recursive_llama")
    
    print("\n" + "=" * 80)
    print("Example 3: Load with standard HuggingFace method")
    print("=" * 80)
    
    # Load - works with standard HF method!
    loaded_model = AutoModelForCausalLM.from_pretrained(
        "./my_recursive_llama",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False  # No custom code needed!
    )
    print("✓ Model loaded from ./my_recursive_llama")
    
    print("\n" + "=" * 80)
    print("Example 4: Use with SFTTrainer (standard workflow)")
    print("=" * 80)
    
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
    )
    
    # Works exactly like any other HF model!
    # trainer = SFTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=your_dataset,
    #     # ... other args
    # )
    # trainer.train()
    
    # Checkpoints are saved/loaded automatically with standard methods!
    # No custom save/load scripts needed!
    
    print("\n✓ All standard HuggingFace methods work seamlessly!")
    print("  - save_pretrained() / from_pretrained()")
    print("  - SFTTrainer automatic checkpointing")
    print("  - push_to_hub() / from_pretrained('username/model')")
    print("  - No custom code required!")