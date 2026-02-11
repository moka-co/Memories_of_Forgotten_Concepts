from datasets import load_dataset

dataset = load_dataset("detection-datasets/coco", trust_remote_code=True)
dataset.save_to_disk("./data/coco")

