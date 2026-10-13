# K9_temperature_scaling.py — paste after K1, K2, K3. No retraining needed.
# Learns a single temperature T per model on val set, reports ECE before/after on test.
# IT FACTOR: most medical-AI papers report accuracy but NOT calibration. We show that
# high accuracy ≠ well-calibrated, and a one-parameter fix (temperature) closes the gap.
# Reliability diagrams before/after = strong visual proof for the paper.

import torch, torch.nn as nn, json, os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

PTH_MAP = [
    ("/kaggle/input/models/nishabhatia122134/stage-1-resnet-us/pytorch/default/1/best.pth",
     "resnet50", 1, "ResNet50_us-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-resnet-none/pytorch/default/1/best.pth",
     "resnet50", 1, "ResNet50_no-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-densenet-us-seed/pytorch/default/1/best.pth",
     "densenet121", 1, "DenseNet121_us-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-densenet-none-seed/pytorch/default/1/best.pth",
     "densenet121", 1, "DenseNet121_no-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-efficient-net-us-seed/pytorch/default/1/best.pth",
     "efficientnet_b0", 1, "EffNet-B0_us-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-eff-none/pytorch/default/1/best.pth",
     "efficientnet_b0", 1, "EffNet-B0_no-aug"),
    ("/kaggle/input/models/nishabhatia122134/stage-1-eff-us-standard/pytorch/default/1/best.pth",
     "efficientnet_b0", 1, "EffNet-B0_standard"),
    ("/kaggle/input/models/nishabhatia122134/stage-2-densenet-us/pytorch/default/1/best.pth",
     "densenet121", 2, "DenseNet121_us-aug_S2"),
    ("/kaggle/input/models/nishabhatia122134/stage-2-eff-us/pytorch/default/1/best.pth",
     "efficientnet_b0", 2, "EffNet-B0_us-aug_S2"),
    ("/kaggle/input/models/nishabhatia122134/stage-2-eff-none/pytorch/default/1/best.pth",
     "efficientnet_b0", 2, "EffNet-B0_no-aug_S2"),
]


@torch.no_grad()
def get_logits(model, X, bs=128):
    model.eval()
    logits = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type=DEVICE.type, enabled=AMP):
            logits.append(model(to_input(X[i:i + bs])).float().cpu())
    return torch.cat(logits)


def learn_temperature(logits, labels, lr=0.01, max_iter=300):
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
    labels_t = torch.tensor(labels, dtype=torch.long)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(logits / T, labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return T.item()


def reliability_diagram(probs, labels, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            bin_acc.append(correct[m].mean())
            bin_conf.append(conf[m].mean())
            bin_count.append(int(m.sum()))
        else:
            bin_acc.append(0.0)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)
    return np.array(bin_acc), np.array(bin_conf), np.array(bin_count), edges


OUT_TS = Path("/kaggle/working/temperature_scaling")
OUT_TS.mkdir(parents=True, exist_ok=True)
rows = []

for pth_path, arch, stage, label in PTH_MAP:
    if not os.path.exists(pth_path):
        print(f"SKIP {label} — {pth_path} not found")
        continue

    root = DATA if stage == 1 else DATA / "brain_subplanes"
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]

    model = build_model(arch, len(classes), 0.3)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = USFolder(root / "val", classes=classes)
    test_ds = USFolder(root / "test", classes=classes)
    Xval, Yval = cache_images(val_ds)
    Xtest, Ytest = cache_images(test_ds)

    val_logits = get_logits(model, Xval)
    test_logits = get_logits(model, Xtest)

    P_before = torch.softmax(test_logits, 1).numpy()
    ece_before = ece_score(P_before, Ytest)
    acc = float((P_before.argmax(1) == Ytest).mean())

    T = learn_temperature(val_logits, Yval)

    P_after = torch.softmax(test_logits / T, 1).numpy()
    ece_after = ece_score(P_after, Ytest)

    improvement = ece_before - ece_after
    print(f"{label:30s} | T={T:.3f} | ECE {ece_before:.4f} -> {ece_after:.4f} "
          f"({'+' if improvement > 0 else ''}{improvement:.4f}) | acc={acc:.4f}")
    rows.append(dict(model=label, stage=stage, temperature=round(T, 4),
                     ece_before=round(ece_before, 4), ece_after=round(ece_after, 4),
                     ece_improvement=round(improvement, 4), acc=round(acc, 4)))

    # reliability diagram: before vs after side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, probs, title in [(axes[0], P_before, "Before (T=1)"),
                             (axes[1], P_after, f"After (T={T:.2f})")]:
        ba, bc, bn, edges = reliability_diagram(probs, Ytest)
        w = np.diff(edges)
        ax.bar(edges[:-1], ba, width=w, align="edge", alpha=0.6, color="#3498db", edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece_score(probs, Ytest):.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(label, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_TS / f"reliability_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    del model, Xval, Xtest, val_logits, test_logits
    torch.cuda.empty_cache()

# ── summary table ──
df_ts = pd.DataFrame(rows)
df_ts.to_csv(OUT_TS / "temperature_scaling_results.csv", index=False)
print("\n" + df_ts.to_string(index=False))

# ── summary bar chart: ECE before vs after ──
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(rows))
ax.bar(x - 0.17, [r["ece_before"] for r in rows], 0.34, label="Before (T=1)",
       color="#e74c3c", alpha=0.85, edgecolor="white")
ax.bar(x + 0.17, [r["ece_after"] for r in rows], 0.34, label="After (T*)",
       color="#2ecc71", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([r["model"] for r in rows], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("ECE (lower = better calibrated)")
ax.set_title("Temperature Scaling: Calibration Improvement Across All Models")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_TS / "ece_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

# ── zip ──
import zipfile
zip_path = "/kaggle/working/TEMPERATURE_SCALING.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT_TS.rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to("/kaggle/working"))
print(f"\nZip: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
from IPython.display import FileLink
FileLink("TEMPERATURE_SCALING.zip")
