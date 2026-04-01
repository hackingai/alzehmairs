"""
setup_dataset.py
----------------
Extracts the Mendeley Alzheimer's Dataset zip, merges train+test
into srikar/dataset/ and wipes the old OASIS images.
"""
import zipfile, os, shutil
from tqdm import tqdm

ZIP_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Alzheimer Dataset.zip")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# wipe old images
print("Clearing old dataset...")
if os.path.isdir(DATASET_DIR):
    for cls in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_path):
            shutil.rmtree(cls_path)

print(f"Extracting: {ZIP_PATH}")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    members = [m for m in z.namelist() if m.lower().endswith(".jpg")]
    for m in tqdm(members, desc="Extracting", unit="img"):
        parts = m.split("/")
        if len(parts) != 4:
            continue
        split = parts[1]   # train / test
        cls   = parts[2]   # MildDemented etc
        fname = f"{split}_{parts[3]}"
        out_dir = os.path.join(DATASET_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, fname)
        with z.open(m) as src, open(dest, "wb") as dst:
            dst.write(src.read())

print("\nClass distribution:")
total = 0
for cls in sorted(os.listdir(DATASET_DIR)):
    p = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(p):
        n = len(os.listdir(p))
        total += n
        print(f"  {cls:<22}: {n} images")
print(f"  {'TOTAL':<22}: {total} images")
print("\nDone. Run: python run_pipeline.py")
