import os, shutil, random
from pathlib import Path
from tqdm import tqdm

random.seed(42)  # for reproducibility

# Input raw dirs (relative paths, since we run from project root)
input_real = "data/raw/real"
input_fake = "data/raw/fake_balanced"

# Output dirs
output_real = "data/real"
output_fake = "data/fake"

splits = {"train": 0.7, "val": 0.15, "test": 0.15}

def split_and_copy(input_dir, output_dir, label):
    files = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
    random.shuffle(files)

    n = len(files)
    train_end = int(n * splits["train"])
    val_end = train_end + int(n * splits["val"])

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_list in split_files.items():
        out_dir = Path(output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(split_list, desc=f"{label}-{split}"):
            shutil.copy(f, out_dir / f.name)

    print(f"✅ {label}: {n} files → train={len(split_files['train'])}, "
          f"val={len(split_files['val'])}, test={len(split_files['test'])}")

# Run for both real and fake
split_and_copy(input_real, output_real, "real")
split_and_copy(input_fake, output_fake, "fake")
