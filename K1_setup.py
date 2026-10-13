# K1_setup.py -- Kaggle notebook, run every session
# Paste this whole file into ONE Kaggle cell and run it.
#
# Kaggle notebook settings (right panel): Accelerator = "GPU T4 x2", Internet = ON
# (Internet is needed once to download the ImageNet-pretrained weights).
# Add data: right panel -> "Add Input" -> Your Datasets -> fetal-planes-final.

import glob
from pathlib import Path
import torch

found = glob.glob("/kaggle/input/**/FETAL_PLANES_DB_data.xlsx", recursive=True)
assert found, "Dataset not found -- add 'fetal-planes-final' with Add Input (right panel)."
DATA = Path(found[0]).parent
for sub in ["train", "val", "test", "brain_subplanes/train", "brain_subplanes/val", "brain_subplanes/test"]:
    assert (DATA / sub).exists(), f"missing folder: {DATA / sub}"

OUT = Path("/kaggle/working/runs")
OUT.mkdir(parents=True, exist_ok=True)

print("DATA:", DATA)
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU -- turn it on in settings")
