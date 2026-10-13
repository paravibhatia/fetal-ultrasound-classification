# 06_augmentation_showcase.py -- run ONCE on Colab
# Paste this whole file into ONE Colab cell and run it.
# Needs: 01_setup.py and 03_augmentation.py run first.
#
# Creates  capstone/augmentation_showcase/  on Drive:
#   augmentation_showcase/
#     Fetal_Brain/
#       Patient00847_Plane3_4_of_4__0_original.png
#       Patient00847_Plane3_4_of_4__1_fan_mask.png
#       Patient00847_Plane3_4_of_4__2_model_input.png
#       Patient00847_Plane3_4_of_4__3_flip.png
#       Patient00847_Plane3_4_of_4__4_rotation_zoom_shift.png
#       Patient00847_Plane3_4_of_4__5_speckle.png
#       Patient00847_Plane3_4_of_4__6_tgc_depth_gain.png
#       Patient00847_Plane3_4_of_4__7_gamma.png
#       Patient00847_Plane3_4_of_4__8_low_resolution.png
#       Patient00847_Plane3_4_of_4__9_combined_random.png
#     ... (10 images x 6 classes, plus the 3 brain sub-planes)
#
# Each transform is shown ON ITS OWN (so you can explain it in the paper / presentation),
# plus one "combined_random" version = what the model actually sees during training.
# Training does NOT use this folder -- training augments on the fly.

import random, shutil
import cv2
import numpy as np

N_PER_CLASS = 10
SHOW = DATA.parent / "augmentation_showcase"
rng = np.random.default_rng(42)
random.seed(42)

def only(img, **force):
    """Apply exactly one transform by switching every other probability to 0 and this one to 1."""
    saved = dict(AUG)
    for k in AUG:
        if k.endswith("_p"):
            AUG[k] = 0.0
    AUG.update(force)
    out = augment(img, "us", rng)
    AUG.clear(); AUG.update(saved)
    return out

if SHOW.exists():
    shutil.rmtree(SHOW)

sources = [(DATA / "train", c.name) for c in sorted((DATA / "train").iterdir()) if c.is_dir()] + \
          [(DATA / "brain_subplanes" / "train", c.name) for c in sorted((DATA / "brain_subplanes" / "train").iterdir()) if c.is_dir()]

total = 0
for root, cls in sources:
    out_dir = SHOW / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in random.sample(sorted((root / cls).glob("*.png")), N_PER_CLASS):
        raw = letterbox(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), 224)
        inp = load_image(path)
        versions = {
            "0_original": raw,
            "1_fan_mask": fan_mask(raw) * 255,
            "2_model_input": inp,
            "3_flip": only(inp, hflip_p=1.0),
            "4_rotation_zoom_shift": only(inp, affine_p=1.0),
            "5_speckle": only(inp, speckle_p=1.0),
            "6_tgc_depth_gain": only(inp, tgc_p=1.0),
            "7_gamma": only(inp, gamma_p=1.0),
            "8_low_resolution": only(inp, degrade_p=1.0),
            "9_combined_random": augment(inp, "us", rng),
        }
        for tag, im in versions.items():
            cv2.imwrite(str(out_dir / f"{path.stem}__{tag}.png"), im)
            total += 1
    print(f"{cls:20s} done")

print(f"\nSaved {total} images to {SHOW}")
