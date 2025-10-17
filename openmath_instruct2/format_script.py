import json

def format_openmath_data(input_file, output_file):
    """
    Reformat OpenMathInstruct-2 data into a simpler format
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
    """
    processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            
            # Create formatted content
            formatted = {
                "content": f"{data['problem']}\n\nThe answer is: \\boxed{{{data['expected_answer']}}}"
            }
            
            outfile.write(json.dumps(formatted) + '\n')
            processed += 1
            
            if processed % 1000 == 0:
                print(f"Processed {processed} samples...")
    
    print(f"\nCompleted! Processed {processed} samples")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    format_openmath_data(
        input_file="/raid/s3/opengptx/behzad_shomali/data/openmath_train_data.jsonl",
        output_file="//raid/s3/opengptx/behzad_shomali/data/openmath_train_formatted.jsonl"
    )

    format_openmath_data(
        input_file="/raid/s3/opengptx/behzad_shomali/data/openmath_test_data.jsonl",
        output_file="//raid/s3/opengptx/behzad_shomali/data/openmath_test_formatted.jsonl"
    )