import yaml

def read_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def generate_config(template, part_num, input_path, output_path, config_output_path):

    new_config = template.copy()
    new_config['settings']['src_path'] = f"{input_path}/Nemotron-CC-Math-v1_train_4plus_part{part_num}.jsonl"
    new_config['settings']['dst_path'] = f"{output_path}/Nemotron-CC-Math-v1_train_4plus_part{part_num}.pbin"
    new_config['settings']['index_path'] = f"{output_path}/Nemotron-CC-Math-v1_train_4plus_part{part_num}.idx"
    new_config['settings']['jq_pattern'] = ".text"
    new_config["tokenizer"]["config"] = {
        "pretrained_model_name_or_path": "/raid/s3/opengptx/behzad_shomali/modalities/llama3-2_1b_tokenizer/",
        "padding": False,
        "truncation": False
    }
    with open(config_output_path.format(part_num=part_num), 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)

def main():
    template_path = "/raid/s3/opengptx/behzad_shomali/modalities/src/modalities/prepare_data/nemtron_24_effective_layers_proj/template.yaml"
    input_path = "/raid/s3/opengptx/behzad_shomali/data/sampled_nvidia___nemotron-cc_JSONL"
    output_path = f"{input_path}/mem_map/nemtron_24_effective_layers_proj"
    config_output_path = "/raid/s3/opengptx/behzad_shomali/modalities/config_files/data_preparation/nemtron_24_effective_layers_proj/Nemotron-CC-highQuality-sampled_data_part{part_num}.yaml"

    template = read_template(template_path)

    for part_num in range(1,25):
       generate_config(template, part_num, input_path, output_path, config_output_path)

    print("Configs generated successfully!")

if __name__ == "__main__":
    main()
