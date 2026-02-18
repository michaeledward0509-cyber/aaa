from datasets import load_dataset
import os
ds = load_dataset("natix-network-org/roadwork", split="train")
DATA_PATH = "/home/dr/myenv/streetvision-subnet/roadwork_data"
for split_name in ["train"]:  # or ["train", "validation"] if the dataset has it
    for label_name, label_id in [("no_roadwork", 0), ("roadwork", 1)]:
        out_dir = os.path.join(DATA_PATH, split_name, label_name)
        os.makedirs(out_dir, exist_ok=True)
    for i, ex in enumerate(ds):
        label_id = ex["label"] if "label" in ex else ex.get("labels", 0)
        label_name = "roadwork" if label_id == 1 else "no_roadwork"
        out_dir = os.path.join(DATA_PATH, "train", label_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{i:06d}.jpg")
        ex["image"].save(path)