import torch
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

# Config
datasets = ["MNIST", "RESISC45", "Cars", "DTD", "EuroSAT", "GTSRB", "SUN397", "SVHN"]
model = "ViT-B-32"
args = parse_arguments()
args.data_location = "/fruitbasket/users/dtredsox/task_vectors/data"
args.model = model
args.cache_dir = "/fruitbasket/users/dtredsox/task_vectors/cache"
args.openclip_cachedir = "/fruitbasket/users/dtredsox/task_vectors/cache"
args.save = f"checkpoints/{model}"
pretrained_checkpoint = f"checkpoints/{model}/zeroshot.pt"

# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f"checkpoints/{model}/{dataset}/finetuned.pt")
    for dataset in datasets
]
# Sum the task vectors
task_vector_sum = sum(task_vectors)
# Apply the resulting task vector
image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0.3)
# Evaluate
for dataset in datasets:
    eval_single_dataset(image_encoder, dataset, args)
