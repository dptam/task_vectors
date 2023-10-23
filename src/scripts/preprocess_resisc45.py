import os
import shutil

resisc_dir = os.path.join("data", "resisc45")


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
        os.path.join(resisc_dir, "resisc45-train.txt")
    ) + get_imageFilepaths(os.path.join(resisc_dir, "resisc45-val.txt"))


def get_valImagesFilepaths():
    return get_imageFilepaths(os.path.join(resisc_dir, "resisc45-test.txt"))


def preprocess_split(split, all_imageFilepaths):
    split_dir = os.path.join(resisc_dir, split)
    safe_makedirs(split_dir)

    # From https://github.com/mlfoundations/task_vectors/issues/1
    for i, image_fp in enumerate(all_imageFilepaths):
        cleaned_imageFilepath = image_fp.strip()
        split_imageFilepath = cleaned_imageFilepath.split("_")
        new_classDirectory = os.path.join(split_dir, "_".join(split_imageFilepath[:-1]))

        safe_makedirs(new_classDirectory)

        original_imageFilepath = os.path.join(
            resisc_dir, "NWPU-RESISC45", "resisc45", cleaned_imageFilepath
        )
        new_imageFilepath = os.path.join(new_classDirectory, split_imageFilepath[-1])
        shutil.copy(original_imageFilepath, new_imageFilepath)

        if i % 100 == 0:
            print(f"Processed {i} / {len(all_imageFilepaths)} images", end="\r")


preprocess_split("train", get_trainImagesFilepaths())
preprocess_split("val", get_valImagesFilepaths())
