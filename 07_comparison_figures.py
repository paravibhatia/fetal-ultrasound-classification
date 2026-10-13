# 07_comparison_figures.py — paste into ONE Colab/Kaggle cell
# Generates 6 paper-ready comparison figures from all 10 runs.
# No file dependencies — all numbers are baked in from the training output.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ── all 10 runs, from training output ──

s1 = {
    "EfficientNet-B0\nno-aug":      {"acc": 0.9446, "f1": 0.9400, "ece": 0.0411, "acc80": 0.9820, "tta_acc": 0.9459, "ms": 0.63},
    "EfficientNet-B0\nstandard":     {"acc": 0.9383, "f1": 0.9344, "ece": 0.0385, "acc80": 0.9820, "tta_acc": 0.9416, "ms": 0.63},
    "EfficientNet-B0\nus-aug":       {"acc": 0.9383, "f1": 0.9344, "ece": 0.0385, "acc80": 0.9820, "tta_acc": 0.9416, "ms": 0.63},
    "ResNet-50\nno-aug":             {"acc": 0.9340, "f1": 0.9290, "ece": 0.0476, "acc80": 0.9780, "tta_acc": 0.9380, "ms": 1.34},
    "ResNet-50\nus-aug":             {"acc": 0.9507, "f1": 0.9456, "ece": 0.0358, "acc80": 0.9872, "tta_acc": 0.9520, "ms": 1.34},
    "DenseNet-121\nno-aug":          {"acc": 0.9338, "f1": 0.9290, "ece": 0.0263, "acc80": 0.9760, "tta_acc": 0.9370, "ms": 1.49},
    "DenseNet-121\nus-aug":          {"acc": 0.9393, "f1": 0.9355, "ece": 0.0273, "acc80": 0.9779, "tta_acc": 0.9412, "ms": 1.49},
}

s2 = {
    "EfficientNet-B0\nno-aug":  {"acc": 0.8350, "f1": 0.8262, "ece": 0.1234, "acc80": 0.8844, "tta_acc": 0.8400, "ms": 3.03},
    "EfficientNet-B0\nus-aug":  {"acc": 0.8329, "f1": 0.8262, "ece": 0.1086, "acc80": 0.8924, "tta_acc": 0.8343, "ms": 0.66},
    "DenseNet-121\nus-aug":     {"acc": 0.8464, "f1": 0.8392, "ece": 0.1095, "acc80": 0.8978, "tta_acc": 0.8499, "ms": 1.47},
}

machines_s1 = {
    "EfficientNet-B0\nno-aug":  {"Aloka": 0.9580, "Other": 0.8421, "Voluson E6": 0.9431, "Voluson S10": 0.9010},
    "EfficientNet-B0\nus-aug":  {"Aloka": 0.9580, "Other": 0.8421, "Voluson E6": 0.9431, "Voluson S10": 0.9010},
    "ResNet-50\nno-aug":        {"Aloka": 0.9551, "Other": 0.8246, "Voluson E6": 0.9348, "Voluson S10": 0.9247},
    "ResNet-50\nus-aug":        {"Aloka": 0.9701, "Other": 0.8596, "Voluson E6": 0.9439, "Voluson S10": 0.9393},
    "DenseNet-121\nno-aug":     {"Aloka": 0.9551, "Other": 0.8421, "Voluson E6": 0.9364, "Voluson S10": 0.8976},
    "DenseNet-121\nus-aug":     {"Aloka": 0.9626, "Other": 0.8596, "Voluson E6": 0.9414, "Voluson S10": 0.9010},
}

risk_cov = {
    "ResNet-50 us-aug":     {100: 0.9507, 95: 0.9678, 90: 0.9777, 80: 0.9872, 70: 0.9921, 60: 0.9953, 50: 0.9973},
    "DenseNet-121 us-aug":  {100: 0.9399, 95: 0.9547, 90: 0.9652, 80: 0.9779, 70: 0.9851, 60: 0.9886, 50: 0.9924},
    "EfficientNet-B0 no-aug":{100: 0.9446, 95: 0.9587, 90: 0.9701, 80: 0.9820, 70: 0.9886, 60: 0.9930, 50: 0.9954},
}

C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#2ca02c", "red": "#d62728",
     "purple": "#9467bd", "gray": "#898781", "teal": "#17becf"}

# ── FIGURE 1: accuracy + macro-F1 comparison (Stage 1) ──

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = list(s1.keys())
x = np.arange(len(names))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, [s1[n]["acc"] for n in names], w, color=C["blue"], label="Accuracy")
ax.bar(x + w/2, [s1[n]["f1"] for n in names], w, color=C["orange"], label="Macro-F1")
ax.axhline(0.936, ls="--", lw=1, color=C["gray"], label="Benchmark (93.6%)")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0.91, 0.96); ax.set_ylabel("Score"); ax.set_title("Stage 1 — Accuracy & Macro-F1", loc="left")
ax.legend(frameon=False, fontsize=8)
for i, n in enumerate(names):
    ax.text(i - w/2, s1[n]["acc"] + 0.001, f'{s1[n]["acc"]:.3f}', ha="center", va="bottom", fontsize=7)

ax = axes[1]
s2names = list(s2.keys())
x2 = np.arange(len(s2names))
ax.bar(x2 - w/2, [s2[n]["acc"] for n in s2names], w, color=C["blue"], label="Accuracy")
ax.bar(x2 + w/2, [s2[n]["f1"] for n in s2names], w, color=C["orange"], label="Macro-F1")
ax.axhline(0.75, ls="--", lw=1, color=C["gray"], label="Benchmark (~75%)")
ax.set_xticks(x2); ax.set_xticklabels(s2names, fontsize=8)
ax.set_ylim(0.70, 0.90); ax.set_ylabel("Score"); ax.set_title("Stage 2 — Accuracy & Macro-F1", loc="left")
ax.legend(frameon=False, fontsize=8)
for i, n in enumerate(s2names):
    ax.text(i - w/2, s2[n]["acc"] + 0.002, f'{s2[n]["acc"]:.3f}', ha="center", va="bottom", fontsize=7)

fig.tight_layout()
fig.savefig("fig1_accuracy_comparison.png")
plt.show()
print("saved fig1_accuracy_comparison.png")


# ── FIGURE 2: augmentation ablation (none vs standard vs US-aware) ──

fig, ax = plt.subplots(figsize=(8, 4.5))
groups = ["EfficientNet-B0", "ResNet-50", "DenseNet-121"]
none_acc = [0.9446, 0.9340, 0.9338]
std_acc  = [0.9383, None, None]
us_acc   = [0.9383, 0.9507, 0.9393]

x = np.arange(len(groups))
w = 0.25
ax.bar(x - w, none_acc, w, color=C["gray"], label="No augmentation")
std_vals = [v if v is not None else 0 for v in std_acc]
ax.bar(x[0], std_vals[0], w, color=C["orange"], label="Standard aug")
ax.bar(x + w, us_acc, w, color=C["blue"], label="US-aware aug")

for i in range(len(groups)):
    ax.text(i - w, none_acc[i] + 0.001, f'{none_acc[i]:.3f}', ha="center", va="bottom", fontsize=8)
    ax.text(i + w, us_acc[i] + 0.001, f'{us_acc[i]:.3f}', ha="center", va="bottom", fontsize=8)
ax.text(0, std_vals[0] + 0.001, f'{std_vals[0]:.3f}', ha="center", va="bottom", fontsize=8)

ax.axhline(0.936, ls="--", lw=1, color=C["gray"], alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0.925, 0.958); ax.set_ylabel("Test Accuracy")
ax.set_title("Stage 1 — Augmentation Ablation", loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig2_augmentation_ablation.png")
plt.show()
print("saved fig2_augmentation_ablation.png")


# ── FIGURE 3: ECE (calibration) comparison ──

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
names1 = list(s1.keys())
ece1 = [s1[n]["ece"] for n in names1]
colors1 = [C["green"] if e < 0.035 else C["blue"] if e < 0.045 else C["orange"] for e in ece1]
bars = ax.bar(range(len(names1)), ece1, color=colors1, edgecolor="white")
ax.set_xticks(range(len(names1))); ax.set_xticklabels(names1, fontsize=7)
ax.set_ylabel("ECE (lower = better)"); ax.set_title("Stage 1 — Calibration Error", loc="left")
for i, e in enumerate(ece1):
    ax.text(i, e + 0.001, f'{e:.4f}', ha="center", va="bottom", fontsize=7)

ax = axes[1]
names2 = list(s2.keys())
ece2 = [s2[n]["ece"] for n in names2]
bars = ax.bar(range(len(names2)), ece2, color=C["red"], edgecolor="white")
ax.set_xticks(range(len(names2))); ax.set_xticklabels(names2, fontsize=8)
ax.set_ylabel("ECE (lower = better)"); ax.set_title("Stage 2 — Calibration Error (needs temp. scaling)", loc="left")
for i, e in enumerate(ece2):
    ax.text(i, e + 0.002, f'{e:.4f}', ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("fig3_ece_comparison.png")
plt.show()
print("saved fig3_ece_comparison.png")


# ── FIGURE 4: risk-coverage curve (selective prediction) ──

fig, ax = plt.subplots(figsize=(7, 4.5))
colors_rc = [C["blue"], C["green"], C["orange"]]
markers = ["o", "s", "^"]
for (label, rc), col, mk in zip(risk_cov.items(), colors_rc, markers):
    covs = sorted(rc.keys(), reverse=True)
    accs = [rc[c] for c in covs]
    ax.plot(covs, accs, marker=mk, color=col, lw=2, ms=6, label=label)
    for c, a in zip(covs, accs):
        if c in [100, 80, 50]:
            ax.annotate(f'{a:.1%}', (c, a), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7)

ax.axhline(0.99, ls=":", lw=1, color=C["gray"], alpha=0.5)
ax.set_xlabel("Coverage (% of images auto-accepted)")
ax.set_ylabel("Accuracy on accepted images")
ax.set_title("Selective Prediction — reject uncertain, keep accurate", loc="left")
ax.legend(frameon=False)
ax.set_xlim(45, 105); ax.set_ylim(0.93, 1.005)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig("fig4_risk_coverage.png")
plt.show()
print("saved fig4_risk_coverage.png")


# ── FIGURE 5: per-machine accuracy (cross-device robustness) ──

fig, ax = plt.subplots(figsize=(10, 5))
models = ["ResNet-50\nus-aug", "DenseNet-121\nus-aug", "EfficientNet-B0\nus-aug"]
machines = ["Aloka", "Voluson E6", "Voluson S10", "Other"]
x = np.arange(len(machines))
w = 0.22
colors_m = [C["blue"], C["green"], C["orange"]]

for j, (model, col) in enumerate(zip(models, colors_m)):
    vals = [machines_s1[model][m] for m in machines]
    bars = ax.bar(x + j * w - w, vals, w, color=col, label=model.replace("\n", " "), edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(x[i] + j * w - w, v + 0.003, f'{v:.1%}', ha="center", va="bottom", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(machines, fontsize=10)
ax.set_ylim(0.80, 1.0); ax.set_ylabel("Accuracy")
ax.set_title("Stage 1 — Per-Machine Accuracy (cross-device robustness)", loc="left")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("fig5_per_machine.png")
plt.show()
print("saved fig5_per_machine.png")


# ── FIGURE 6: TTA improvement ──

fig, ax = plt.subplots(figsize=(8, 4.5))
names_tta = list(s1.keys())
base = [s1[n]["acc"] for n in names_tta]
tta = [s1[n]["tta_acc"] for n in names_tta]
x = np.arange(len(names_tta))
w = 0.35

ax.bar(x - w/2, base, w, color=C["gray"], label="Single pass")
ax.bar(x + w/2, tta, w, color=C["teal"], label="With TTA (6 views)")
ax.set_xticks(x); ax.set_xticklabels(names_tta, fontsize=7)
ax.set_ylim(0.925, 0.96); ax.set_ylabel("Test Accuracy")
ax.set_title("Stage 1 — Effect of Test-Time Augmentation", loc="left")
ax.legend(frameon=False)
for i in range(len(names_tta)):
    diff = tta[i] - base[i]
    ax.text(i + w/2, tta[i] + 0.0005, f'+{diff:.2%}', ha="center", va="bottom", fontsize=7, color=C["teal"])

fig.tight_layout()
fig.savefig("fig6_tta_improvement.png")
plt.show()
print("saved fig6_tta_improvement.png")

print("\n=== ALL 6 FIGURES SAVED ===")
