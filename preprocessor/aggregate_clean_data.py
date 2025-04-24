import os
import json
from collections import Counter
import argparse

def save_preprocessed_data(clean_data,output_file):

    updated_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            updated_data = json.load(f)

    updated_data.extend(clean_data)

    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=2)


def remap_data(data_file_path, class_map_path,output_file_path):
    with open(data_file_path, 'r') as f:
        clean_data = json.load(f)

    with open(class_map_path, 'r') as f:
        class_map = json.load(f)

    # for each sample add label_class name
    for sample in clean_data:
        sample['label_class'] = class_map[sample['label']]

    # count unique labels in the filtered_data
    label_counts = Counter(sample['label'] for sample in clean_data)

    save_preprocessed_data(clean_data,output_file_path)

    print(f"Preprocessing complete: {len(clean_data)} samples with {len(label_counts)} remapped labels.")

def main():
    parser = argparse.ArgumentParser(description="Remap MS-ASL dataset videos.")
    parser.add_argument('--data_dir', type=str, default="../Data/MS-ASL-Clean-Data", help="Path to cleaned MS-ASL data")
    parser.add_argument('--class_map_path', type=str, default="../Data/MS-ASL/MSASL_classes.json",
                        help="Path to class map")
    parser.add_argument('--output_file', type=str, default="clean_data.json",
                        help="Output file name for preprocessed data")

    args = parser.parse_args()

    clean_data_dir = args.data_dir
    class_map_path = args.class_map_path
    output_file = args.output_file

    for filename in ["train.json", "val.json", "test.json"]:
        if filename.endswith(".json"):
            data_file_path = os.path.join(clean_data_dir, filename)
            output_file_path = os.path.join(clean_data_dir, output_file)
            remap_data(data_file_path, class_map_path,output_file_path)

if __name__ == "__main__":
    main()