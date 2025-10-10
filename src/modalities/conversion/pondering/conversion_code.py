"""Code transfer utilities for Pondering model conversion."""

import os
import shutil


def _copy_model_files(output_dir: str):
    """Copy model files to output directory."""
    source_dir = os.path.dirname(__file__)
    
    files_to_copy = [
        "modeling_pondering.py",
        "configuration_pondering.py",
    ]
    
    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, output_dir)


def _change_modalities_import_to_relative_import(output_dir: str):
    """Change modalities imports to relative imports."""
    target_modeling_file = os.path.join(output_dir, "modeling_gpt2.py")
    
    if not os.path.exists(target_modeling_file):
        return
    
    with open(target_modeling_file, "r") as file:
        content = file.read()
    
    # Replace modalities imports with relative imports
    content = content.replace(
        "from modalities.conversion.pondering.configuration_pondering",
        "from .configuration_pondering"
    )
    content = content.replace(
        "from modalities.conversion.gpt2.modeling_gpt2",
        "from .modeling_gpt2"
    )
    
    with open(target_modeling_file, "w") as file:
        file.write(content)


def _copy_base_model_files(output_dir: str):
    """Copy base model files (GPT2) to output directory."""
    source_dir = os.path.dirname(__file__)
    gpt2_source_dir = os.path.join(os.path.dirname(source_dir), "gpt2")
    pondering_source_dir = os.path.join(os.path.dirname(source_dir), "pondering")
    
    base_model_files = [
        "modeling_gpt2.py",
        "configuration_gpt2.py",
    ]
    
    # for filename in base_model_files:
    source_path = os.path.join(gpt2_source_dir, "modeling_gpt2.py")
    additional_path = os.path.join(pondering_source_dir, "configuration_pondering.py")
    
    if os.path.exists(source_path) and os.path.exists(additional_path):
        # create a new file called modeling_gpt2.py in output_dir whihc is a copy of source_path appended with the contents of additional_path
        with open(source_path, "r") as src_file:
            src_content = src_file.read()
        with open(additional_path, "r") as add_file:
            add_content = add_file.read()
        with open(os.path.join(output_dir, "modeling_gpt2.py"), "w") as dest_file:
            dest_file.write(src_content + "\n\n" + add_content)
        
        gpt_configuration_path = os.path.join(gpt2_source_dir, "configuration_gpt2.py")
        shutil.copy(gpt_configuration_path, output_dir)


def _change_gpt2_imports_to_relative(output_dir: str):
    """Change GPT2 modalities imports to relative imports."""
    target_file = os.path.join(output_dir, "modeling_gpt2.py")
    
    if not os.path.exists(target_file):
        return
    
    with open(target_file, "r") as file:
        content = file.read()
    
    content = content.replace(
        "from modalities.conversion.gpt2.configuration_gpt2",
        "from .configuration_gpt2"
    )
    
    with open(target_file, "w") as file:
        file.write(content)


def transfer_model_code(output_dir: str):
    """
    Copies the required model code to the output directory and replaces modalities imports.
    This allows the converted model to be used without the modalities package via:
    
    >>> from transformers import AutoModelForCausalLM
    >>> model = AutoModelForCausalLM.from_pretrained("path/to/converted/model", trust_remote_code=True)

    Args:
        output_dir (str): Directory of the converted model.
    """     
    # Copy pondering model files
    _copy_model_files(output_dir)
    
    # Copy base model (GPT2) files
    _copy_base_model_files(output_dir)
    
    # Fix imports in pondering model
    _change_modalities_import_to_relative_import(output_dir)
    
    # Fix imports in base model
    _change_gpt2_imports_to_relative(output_dir)