"""
extract_and_prepare.py
----------------------
Extracts all OASIS oasis_cross-sectional_disc*.tar.gz files in the srikar folder,
reads each subject's CDR score, and organises the MRI preview GIF images into
the dataset folder structure expected by data_pipeline.py / train.py:

  dataset/
    NonDemented/          CDR == 0
    VeryMildDemented/     CDR == 0.5
    MildDemented/         CDR == 1
    ModerateDemented/     CDR >= 2

Usage:
    python extract_and_prepare.py
    python extract_and_prepare.py --src . --out ./dataset --extract-dir ./extracted
"""

import argparse
import glob
import io
import os
import re
import shutil
import tarfile
from tqdm import tqdm

# CDR -> class folder mapping (matches data_pipeline.py CLASSES)
CDR_TO_CLASS = {
    0.0:  "NonDemented",
    0.5:  "VeryMildDemented",
    1.0:  "MildDemented",
    2.0:  "ModerateDemented",
    3.0:  "SevereDemented",
}


def parse_cdr(txt_bytes: bytes) -> "float | None":
    """Extract CDR value from a subject metadata .txt file."""
    text = txt_bytes.decode(errors="ignore")
    match = re.search(r"CDR:\s*([\d.]+)", text)
    if match:
        return float(match.group(1))
    return None


def process_archive(tar_path: str, dataset_dir: str, extract_dir: str, stats: dict):
    """Extract one tar.gz, read CDR labels, copy GIF slices to dataset folders."""
    print(f"\n[+] Processing: {os.path.basename(tar_path)}")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()

        # Group members by subject session ID
        subjects: dict[str, list] = {}
        for m in members:
            parts = m.name.split("/")
            if len(parts) >= 2:
                session = parts[1]  # e.g. OAS1_0001_MR1
                subjects.setdefault(session, []).append(m)

        for session, session_members in tqdm(subjects.items(), desc=f"  {os.path.basename(tar_path)[:28]}", unit="subj", leave=False, colour="green"):
            # Find the metadata .txt for this session
            txt_member = next(
                (m for m in session_members
                 if m.name.endswith(f"{session}.txt") and "FSL_SEG" not in m.name),
                None,
            )
            if txt_member is None:
                print(f"  [!] No metadata txt for {session}, skipping")
                stats["skipped"] += 1
                continue

            cdr = parse_cdr(tf.extractfile(txt_member).read())
            if cdr is None:
                print(f"  [!] CDR not found for {session}, skipping")
                stats["skipped"] += 1
                continue

            class_name = CDR_TO_CLASS.get(cdr)
            if class_name is None:
                print(f"  [!] Unknown CDR={cdr} for {session}, skipping")
                stats["skipped"] += 1
                continue

            # Collect GIF preview images (processed T88 coronal slices preferred)
            gif_members = [
                m for m in session_members
                if m.name.endswith(".gif") and "PROCESSED" in m.name
            ]
            # Fallback to RAW gifs if no processed ones
            if not gif_members:
                gif_members = [m for m in session_members if m.name.endswith(".gif")]

            if not gif_members:
                print(f"  [!] No GIF images for {session}, skipping")
                stats["skipped"] += 1
                continue

            out_dir = os.path.join(dataset_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)

            for gif_m in gif_members:
                fname = os.path.basename(gif_m.name)
                dest = os.path.join(out_dir, f"{session}_{fname}")
                if os.path.exists(dest):
                    continue  # already extracted
                data = tf.extractfile(gif_m)
                if data:
                    with open(dest, "wb") as f:
                        f.write(data.read())
                    stats["copied"] += 1

            stats["subjects"] += 1
            print(f"  CDR={cdr:>4}  ->  {class_name:<20}  ({len(gif_members)} images)  [{session}]")


def print_summary(dataset_dir: str, stats: dict):
    print("\n" + "=" * 55)
    print("Extraction complete")
    print(f"  Subjects processed : {stats['subjects']}")
    print(f"  Images copied      : {stats['copied']}")
    print(f"  Skipped            : {stats['skipped']}")
    print("\nDataset class distribution:")
    for cls in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented", "SevereDemented"]:
        path = os.path.join(dataset_dir, cls)
        count = len(os.listdir(path)) if os.path.isdir(path) else 0
        print(f"  {cls:<22}: {count} images")
    print("=" * 55)
    print(f"\nDataset ready at: {os.path.abspath(dataset_dir)}")
    print("Run training with:")
    print(f"  python train.py --dataset {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract OASIS tar.gz and build dataset")
    parser.add_argument("--src", default=".", help="Folder containing the .tar.gz files (default: .)")
    parser.add_argument("--out", default="./dataset", help="Output dataset directory (default: ./dataset)")
    parser.add_argument("--extract-dir", default="./extracted", help="Temp extraction directory (unused, kept for compat)")
    args = parser.parse_args()

    archives = sorted(glob.glob(os.path.join(args.src, "oasis_cross-sectional_disc*.tar.gz")))
    if not archives:
        print(f"No oasis_cross-sectional_disc*.tar.gz files found in: {args.src}")
        return

    print(f"Found {len(archives)} archive(s) in '{args.src}'")
    print(f"Output dataset dir: {args.out}\n")

    stats = {"subjects": 0, "copied": 0, "skipped": 0}

    for archive in tqdm(archives, desc="Archives", unit="file", colour="cyan"):
        process_archive(archive, args.out, args.extract_dir, stats)

    print_summary(args.out, stats)


if __name__ == "__main__":
    main()
