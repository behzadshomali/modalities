import os

def split_jsonl_file(input_filename, num_files):
    """
    Reads a JSONL file and splits it into a specified number of smaller files.

    Args:
        input_filename (str): The name of the large JSONL file to split.
        num_files (int): The number of smaller files to create.
    """
    # 1. Get the total number of lines in the input file
    print(f"Counting lines in {input_filename}...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return

    if total_lines == 0:
        print("The input file is empty.")
        return

    # 2. Calculate the number of lines per file
    # Use integer division (//) for the base line count.
    # The remainder (%) will be distributed to the first few files.
    lines_per_file = total_lines // num_files
    remainder = total_lines % num_files
    print(f"Total lines: {total_lines}")
    print(f"Base lines per file: {lines_per_file}")
    print(f"Remainder: {remainder}")

    # 3. Create a list of the exact line counts for each output file
    # The first 'remainder' files get an extra line.
    file_line_counts = [lines_per_file + 1 if i < remainder else lines_per_file for i in range(num_files)]
    print(f"Target line counts: {file_line_counts}")

    # 4. Iterate through the input file and write to the output files
    print("Starting file splitting...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            line_index = 0
            for i in range(num_files):
                # Construct the output filename (e.g., 'data_part_0.jsonl')
                output_filename = f"{os.path.splitext(input_filename)[0]}_part_{i}.jsonl"
                
                # Get the target number of lines for the current file
                current_file_lines = file_line_counts[i]
                
                print(f"Writing {current_file_lines} lines to {output_filename}")
                
                with open(output_filename, 'w', encoding='utf-8') as outfile:
                    lines_written = 0
                    while lines_written < current_file_lines:
                        line = infile.readline()
                        if not line:
                            # Should not happen if line count was correct, but good to handle
                            break
                        outfile.write(line)
                        lines_written += 1
                        line_index += 1

        print("\n✅ Splitting complete!")
        print(f"Original file '{input_filename}' split into {num_files} parts.")

    except Exception as e:
        print(f"\nAn error occurred during splitting: {e}")

BIG_FILE_NAME = "/raid/s3/opengptx/behzad_shomali/data/sampled_nvidia___nemotron-cc_JSONL/Nemotron-CC-highQuality-sampled_data_part_23_part_0_SMALL_EVAL.jsonl"
NUMBER_OF_SPLIT_FILES = 60

split_jsonl_file(BIG_FILE_NAME, NUMBER_OF_SPLIT_FILES)