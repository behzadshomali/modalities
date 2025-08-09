import yaml

def read_template(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def generate_config(template, language, input_path, output_path):
    for split in ['train', 'test', 'val']:
        new_config = template.copy()
        new_config['settings']['src_path'] = f"{input_path}/multilingual_fineweb_{language}_{split}.jsonl"
        new_config['settings']['dst_path'] = f"{output_path}/multilingual_fineweb_{language}_{split}.pbin"
        new_config['settings']['index_path'] = f"{output_path}/multilingual_fineweb_{language}_{split}.idx"

        with open(f"modalities/src/modalities/multilingual_proj/configs/multilingual_fineweb_{language}_{split}_dataset_config.yaml", 'w') as file:
            yaml.dump(new_config, file, default_flow_style=False)

def main():
    template_path = "modalities/src/modalities/multilingual_proj/configs/dataset_config_template.yaml"
    input_path = "/raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/texts/splitted"
    output_path = "/raid/s3/opengptx/behzad_shomali/multilingual_llm_proj/mem_map"

    template = read_template(template_path)

    for language in ['en', 'fra_Latn', 'deu_Latn', 'ita_Latn', 'spa_Latn']:
       generate_config(template, language, input_path, output_path)

if __name__ == "__main__":
    main()
