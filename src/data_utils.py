"""Download the dataset, convert MATLAB files to PNG, split into train/val/test."""
import json
import urllib.request
import zipfile

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config as C


def download_dataset():
    """Download the 4 zip files from Figshare (~880 MB) and unzip them."""
    if len(list((C.RAW_DIR / "mat").glob("*.mat"))) == 3064:
        print("dataset already downloaded")
        return
    with urllib.request.urlopen(C.FIGSHARE_API) as r:
        files = [f for f in json.load(r)["files"] if f["name"].endswith(".zip")]
    for f in files:
        zip_path = C.RAW_DIR / f["name"]
        if not zip_path.exists():
            print("downloading", f["name"], f"({f['size'] / 1e6:.0f} MB) ...")
            urllib.request.urlretrieve(f["download_url"], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(C.RAW_DIR / "mat")
    print(len(list((C.RAW_DIR / "mat").glob("*.mat"))), ".mat files ready")


def read_mat(path):
    """One .mat file = one MRI slice + tumour mask + label + patient id."""
    with h5py.File(path, "r") as f:
        d = f["cjdata"]
        image = np.array(d["image"], dtype=np.float32).T
        mask = np.array(d["tumorMask"], dtype=np.uint8).T
        label = int(np.array(d["label"]).squeeze())
        pid = "".join(chr(int(c)) for c in np.array(d["PID"]).squeeze())
    return image, mask, label, pid


def convert_to_png(size=512):
    """Save every slice as an 8-bit PNG and build a table with the labels."""
    rows = []
    mat_files = sorted((C.RAW_DIR / "mat").glob("*.mat"), key=lambda p: int(p.stem))
    for p in tqdm(mat_files):
        image, mask, label, pid = read_mat(p)
        # MRI intensities are not 0-255 -> rescale to 8 bit
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = cv2.resize(image.astype(np.uint8), (size, size))
        mask = cv2.resize(mask * 255, (size, size), interpolation=cv2.INTER_NEAREST)
        name = f"{int(p.stem):04d}.png"
        cv2.imwrite(str(C.IMAGES_DIR / name), image)
        cv2.imwrite(str(C.MASKS_DIR / name), mask)
        rows.append({"file": name, "patient": pid, "tumor_type": C.TUMOR_TYPES[label],
                     "label": C.TUMOR_TO_CLASS[label]})
    df = pd.DataFrame(rows)
    df["class"] = df["label"].map(lambda l: C.CLASS_NAMES[l])
    df.to_csv(C.SPLITS_DIR / "all_images.csv", index=False)
    return df


def make_splits(df, images_per_class=C.IMAGES_PER_CLASS,
                max_per_patient=C.MAX_SLICES_PER_PATIENT, seed=C.SEED):
    """Pick a small subset and split it 70 / 15 / 15 **by patient**.

    * All slices of one patient go to the same split. Slices of the same patient
      look almost the same, so mixing them would give a fake high test accuracy.
    * At most `max_per_patient` slices per patient -> more different patients.
    * Each class is split separately, so every split has both classes."""
    rng = np.random.default_rng(seed)
    parts = []
    for label in (0, 1):
        cls = df[df.label == label]
        patients = rng.permutation(cls.patient.unique().tolist())
        chosen, count = [], 0
        for pat in patients:
            if images_per_class and count >= images_per_class:
                break
            slices = cls[cls.patient == pat]
            slices = slices.sample(min(len(slices), max_per_patient), random_state=seed)
            chosen.append(slices)
            count += len(slices)
        sub = pd.concat(chosen)

        pats = sub.patient.unique()          # already in random order
        n = len(pats)
        split_of = {p: "train" for p in pats[: int(0.70 * n)]}
        split_of.update({p: "val" for p in pats[int(0.70 * n): int(0.85 * n)]})
        split_of.update({p: "test" for p in pats[int(0.85 * n):]})
        sub["split"] = sub.patient.map(split_of)
        parts.append(sub)
    sub = pd.concat(parts)

    sub.to_csv(C.SPLITS_DIR / "dataset.csv", index=False)
    for s in ("train", "val", "test"):
        sub[sub.split == s].to_csv(C.SPLITS_DIR / f"{s}.csv", index=False)
    return sub
