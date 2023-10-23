import os
import shutil

dtd_dir = os.path.join("data", "dtd")


def get_imageFilepaths(split_filepath):
    all_imageFilepaths = []
    with open(split_filepath, "r+") as f:
        for line in f.readlines():
            all_imageFilepaths.append(line)
    return all_imageFilepaths


def safe_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_trainImagesFilepaths():
    return get_imageFilepaths(
        os.path.join(dtd_dir, "labels", "train1.txt")
    ) + get_imageFilepaths(os.path.join(dtd_dir, "labels", "val1.txt"))


def get_valImagesFilepaths():
    return get_imageFilepaths(os.path.join(dtd_dir, "labels", "test1.txt"))


def preprocess_split():
    # split = "train"
    # all_imageFilepaths = get_trainImagesFilepaths()
    split = "val"
    all_imageFilepaths = get_valImagesFilepaths()

    split_dir = os.path.join(dtd_dir, split)
    safe_makedirs(split_dir)

    # From https://github.com/mlfoundations/task_vectors/issues/1
    for i, image_fp in enumerate(all_imageFilepaths):
        cleaned_imageFilepath = image_fp.strip()
        split_imageFilepath = cleaned_imageFilepath.split("/")
        new_classDirectory = os.path.join(split_dir, *split_imageFilepath[:-1])

        safe_makedirs(new_classDirectory)

        original_imageFilepath = os.path.join(dtd_dir, "images", cleaned_imageFilepath)
        new_imageFilepath = os.path.join(new_classDirectory, split_imageFilepath[-1])

        shutil.copy(original_imageFilepath, new_imageFilepath)

        if i % 100 == 0:
            print(f"Processed {i} / {len(all_imageFilepaths)} images", end="\r")


preprocess_split()
