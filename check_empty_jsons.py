import os
import json

# Define the base directory where the train, val, test folders reside
base_dataset_dir = "/Users/doodledaron/Documents/Freelances/Leon/law bundling POC/CUAD_v1/layoutlmv3_dataset_part1"
splits = ["train", "val", "test"]

print(f"Checking for empty JSON files in: {base_dataset_dir}")
print("-" * 30)

# Iterate over each split directory (train, val, test)
for split in splits:
    split_dir = os.path.join(base_dataset_dir, split)
    empty_json_count = 0
    total_json_count = 0

    if not os.path.isdir(split_dir):
        print(f"Directory not found: {split_dir}")
        continue

    # List all files in the current split directory
    for filename in os.listdir(split_dir):
        # Process only JSON files, skip the label map
        if filename.endswith(".json") and filename != "label_map.json":
            total_json_count += 1
            file_path = os.path.join(split_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read content, strip whitespace to check for truly empty files
                    content = f.read().strip()
                    if not content: # Check if file is completely empty or just whitespace
                         empty_json_count += 1
                         print(f"Found empty file: {file_path}") # <<< UNCOMMENTED THIS LINE >>>
                         continue

                    # Try to parse the JSON content
                    data = json.loads(content) # Use loads on the read content

                    # Check if the parsed data is an empty list
                    if isinstance(data, list) and not data:
                        empty_json_count += 1
                        print(f"Found empty list JSON: {file_path}") # <<< UNCOMMENTED THIS LINE >>>

            except json.JSONDecodeError:
                # Count files that are not valid JSON
                empty_json_count += 1
                print(f"Found invalid JSON: {file_path}") # <<< UNCOMMENTED THIS LINE >>>
            except Exception as e:
                # Catch other potential errors during file processing
                print(f"Error processing file {file_path}: {e}")
                empty_json_count += 1 # Treat errors as potentially problematic/empty

    # Print the result for the current split
    print(f"{split.capitalize()} directory:")
    print(f"  Total JSON files found: {total_json_count}")
    print(f"  Empty or invalid JSON files: {empty_json_count}")
    print("-" * 30)

print("Check complete.")