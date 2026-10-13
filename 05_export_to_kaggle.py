# 05_export_to_kaggle.py -- run ONCE on Colab
# Paste this whole file into ONE Colab cell and run it.
# Needs: 01_setup.py run first.
#
# What it does:
#   1. Makes a compact copy of dataset_final (train/val/test + brain_subplanes) with the
#      longest side shrunk to 384 px. The model uses 224 px, so no information it uses is lost,
#      but the upload is ~5x smaller and Kaggle training reads images much faster.
#   2. Uploads it to Kaggle as a PRIVATE dataset called "fetal-planes-final" (server-to-server,
#      nothing is downloaded to your laptop).
#
# Before running: kaggle.com -> your profile picture -> Settings -> API -> "Create New Token".
# That downloads kaggle.json. This cell will ask you to upload that file.

import os, json, shutil, cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

MAX_SIDE = 384
LOCAL = Path("/content/kaggle_upload")
OUT = LOCAL / "dataset_final"
SUBSETS = ["train", "val", "test", "brain_subplanes/train", "brain_subplanes/val", "brain_subplanes/test"]

# ---------- 1. compact copy ----------
if LOCAL.exists():
    shutil.rmtree(LOCAL)
jobs = [(p, OUT / sub / p.parent.name / p.name) for sub in SUBSETS for p in (DATA / sub).glob("*/*.png")]
print("images to copy:", len(jobs))

def shrink(job):
    src, dst = job
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst), img)

with ThreadPoolExecutor(16) as ex:
    ok = sum(tqdm(ex.map(shrink, jobs), total=len(jobs)))
print("written:", ok, "/", len(jobs))

shutil.copy(DATA.parent / "FETAL_PLANES_DB_data.xlsx", OUT / "FETAL_PLANES_DB_data.xlsx")
shutil.copy(DATA / "val_patients.csv", OUT / "val_patients.csv")

for sub in SUBSETS:
    counts = {c.name: len(list(c.glob("*.png"))) for c in sorted((OUT / sub).iterdir()) if c.is_dir()}
    print(f"{sub:24s} total={sum(counts.values()):5d}  {counts}")

# ---------- 2. upload to Kaggle ----------
from google.colab import files
print("\nUpload your kaggle.json now:")
uploaded = files.upload()
token = json.loads(list(uploaded.values())[0])
for d in [Path.home() / ".kaggle", Path.home() / ".config" / "kaggle"]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "kaggle.json").write_text(json.dumps(token))
    os.chmod(d / "kaggle.json", 0o600)

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

meta = {"title": "fetal-planes-final", "id": f"{token['username']}/fetal-planes-final",
        "licenses": [{"name": "other"}]}
(LOCAL / "dataset-metadata.json").write_text(json.dumps(meta))

print("Uploading (a few minutes)...")
api.dataset_create_new(folder=str(LOCAL), public=False, quiet=False, dir_mode="zip")
print(f"\nDONE. Check https://www.kaggle.com/datasets/{token['username']}/fetal-planes-final "
      "(Kaggle needs a few minutes to finish processing).")
