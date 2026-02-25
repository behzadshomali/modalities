from transformers import LlamaConfig

class RecursiveLlamaConfig(LlamaConfig):
    model_type = "recursive_llama"
    
    def __init__(
        self,
        recursive_start_layer=None,
        recursive_end_layer=None,
        num_recursions=3,
        sample_random_recursion=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.recursive_start_layer = recursive_start_layer
        self.recursive_end_layer = recursive_end_layer
        self.num_recursions = num_recursions
        self.sample_random_recursion = sample_random_recursion