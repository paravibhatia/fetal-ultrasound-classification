# 04_preview.py -- augmentation preview
# Paste this whole file into ONE Colab cell and run it.
# Needs: 01_setup.py and 03_augmentation.py run first.

import random
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
random.seed(7)
classes = sorted(c.name for c in (DATA / "train").iterdir() if c.is_dir())
titles = ["original", "fan mask", "model input", "standard aug", "US aug 1", "US aug 2", "US aug 3", "US aug 4"]

fig, axes = plt.subplots(len(classes), len(titles), figsize=(2.2 * len(titles), 2.3 * len(classes)))
for r, cls in enumerate(classes):
    path = random.choice(list((DATA / "train" / cls).glob("*.png")))
    raw = letterbox(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), 224)
    inp = load_image(path)
    panels = [raw, fan_mask(raw) * 255, inp, augment(inp, "standard", rng)] + [augment(inp, "us", rng) for _ in range(4)]
    for c, im in enumerate(panels):
        axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0: axes[r, c].set_title(titles[c], fontsize=9)
        if c == 0: axes[r, c].set_ylabel(cls, fontsize=9)
plt.tight_layout()
plt.savefig(DATA.parent / "augmentation_preview.png", dpi=130)
plt.show()
