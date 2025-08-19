import os
import re

# Directory containing your files
directory = "/raid/s3/opengptx/behzad_shomali/checkpoints/2025-08-11__13-49-24_a0c553c2"

# Regex to extract the seen_tokens value
pattern = re.compile(r"seen_steps_(\d+)")

# Iterate through files and filter
l = []
for filename in os.listdir(directory):
    match = pattern.search(filename)
    if match:
        seen_steps = int(match.group(1))
        if seen_steps % 2048 == 0 and seen_steps > 76800:
            pass
            # l.append(seen_steps)
            # print(filename)
        else:
            # print(filename)
            # remove the file
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed: {filename}")

print(sorted(l))