import os
import shutil
import random

# From https://github.com/mlfoundations/task_vectors/issues/1


def safe_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_directory_structure(base_dir, classes):
    for dataset in ["train", "val", "test"]:
        path = os.path.join(base_dir, dataset)
        for cls in classes:
            safe_makedirs(os.path.join(path, cls))


def split_dataset(base_dir, source_dir, classes, val_size=270, test_size=270):
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size : val_size + test_size]
        train_images = images[val_size + test_size :]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, "train", class_name, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, "val", class_name, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, "test", class_name, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)


source_dir = "/fruitbasket/users/dtredsox/task_vectors/data/EuroSAT_splits/EuroSAT_RGB"  # replace with the path to your dataset
base_dir = "/fruitbasket/users/dtredsox/task_vectors/data/EuroSAT_splits/"  # replace with the path to the output directory

classes = [
    class_name
    for class_name in os.listdir(source_dir)
    if os.path.isdir(os.path.join(source_dir, class_name))
]

create_directory_structure(base_dir, classes)
split_dataset(base_dir, source_dir, classes)
