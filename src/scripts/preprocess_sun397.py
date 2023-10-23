## PROCESS SUN397 DATASET

import os
import shutil
from pathlib import Path

# From https://github.com/mlfoundations/task_vectors/issues/1


def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split("/")[:-1])[1:]
        filename = input_path.split("/")[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)

        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")


data_dir = "/fruitbasket/users/dtredsox/task_vectors/data/sun397"
# process_dataset(
#     os.path.join(data_dir, "partitions", "Training_01.txt"),
#     os.path.join(data_dir, "SUN397"),
#     os.path.join(data_dir, "train"),
# )
process_dataset(
    os.path.join(data_dir, "partitions", "Testing_01.txt"),
    os.path.join(data_dir, "SUN397"),
    os.path.join(data_dir, "val"),
)
