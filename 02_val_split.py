# 02_val_split.py -- run ONCE (dry run first, then DRY_RUN = False)
# Paste this whole file into ONE Colab cell and run it.
# Needs: 01_setup.py run first.

# Moves 12% of TRAIN patients into val/ (same patients for brain_subplanes).
# Run with DRY_RUN = True first, check every class has val images, then set False and run again.
import re, random, shutil, csv
from collections import Counter

DRY_RUN = True
VAL_FRAC, SEED = 0.12, 42

def patient_of(name):
    return re.search(r"(Patient\d+)", name).group(1)

def list_images(split_dir):
    return [(c.name, f) for c in sorted(split_dir.iterdir()) if c.is_dir()
            for f in c.iterdir() if f.suffix.lower() == ".png"]

if (DATA / "val").exists() and any((DATA / "val").rglob("*.png")):
    raise SystemExit("val/ already exists -- split was already done, don't run again.")

s1 = list_images(DATA / "train")
patients = sorted({patient_of(f.name) for _, f in s1})
random.Random(SEED).shuffle(patients)
val_patients = set(patients[:round(VAL_FRAC * len(patients))])
print(f"train patients: {len(patients)} -> val patients: {len(val_patients)}")

def move(images, val_root, title):
    kept, moved = Counter(), Counter()
    for cls, f in images:
        if patient_of(f.name) in val_patients:
            moved[cls] += 1
            if not DRY_RUN:
                (val_root / cls).mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(val_root / cls / f.name))
        else:
            kept[cls] += 1
    print(f"\n{title}   class: train / val")
    for c in sorted(kept | moved):
        print(f"  {c:20s} {kept[c]:5d} / {moved[c]:4d}")

move(s1, DATA / "val", "STAGE 1")
move(list_images(DATA / "brain_subplanes" / "train"), DATA / "brain_subplanes" / "val", "STAGE 2")

if not DRY_RUN:
    with open(DATA / "val_patients.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["patient_id"]] + [[p] for p in sorted(val_patients)])
    print("\nDone. Saved val_patients.csv")
else:
    print("\nDRY RUN only -- nothing moved. Set DRY_RUN = False and run again.")
