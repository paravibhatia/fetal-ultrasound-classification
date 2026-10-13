# K10_leave_machine_out.py — paste after K1, K2, K3. RETRAINS models (needs GPU).
# For each ultrasound machine, train ResNet-50 on the OTHER machines, test on the held-out one.
# IT FACTOR: tests whether the model learns ANATOMY or MACHINE-SPECIFIC artifacts.
# If accuracy crashes for one machine → that machine has unique texture the model relied on.
# Directly relevant to clinical deployment — hospitals use different US hardware.
# Most papers skip this. We show robustness (or lack of it) to hardware domain shift.

import time, json, copy, random, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

machines = sorted(set(MACHINE_OF.values()))
print(f"Machines found: {machines}")
print(pd.Series(MACHINE_OF.values()).value_counts().to_string())


class MachineFilteredDataset:
    """USFolder filtered to include/exclude specific ultrasound machines."""
    def __init__(self, root, classes, include=None, exclude=None, aug=None, size=224):
        root = Path(root)
        self.classes, self.aug, self.size = classes, aug, size
        self.samples = []
        for i, c in enumerate(classes):
            cls_dir = root / c
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.glob("*.png")):
                m = MACHINE_OF.get(f.stem, "?")
                if include and m not in include:
                    continue
                if exclude and m in exclude:
                    continue
                self.samples.append((str(f), i))
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = load_image(path, self.size)
        if self.aug:
            img = augment(img, self.aug, np.random.default_rng())
        return torch.from_numpy(img), y, i


OUT_LMO = Path("/kaggle/working/leave_machine_out")
OUT_LMO.mkdir(parents=True, exist_ok=True)

arch, aug_mode = "resnet50", "us"
epochs, patience = 25, 6
bs, lr, wd, dropout, seed = 32, 2e-4, 1e-4, 0.3, 42
root = DATA
stage1_classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
print(f"Classes: {stage1_classes}")

# also get baseline accuracy (all machines) from saved results if available
baseline_acc = 0.9507  # ResNet-50 us-aug from training runs

results_lmo = []

for held_out in machines:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'=' * 70}")
    print(f"LEAVE OUT: {held_out}")
    print(f"{'=' * 70}")

    train_ds = MachineFilteredDataset(root / "train", stage1_classes,
                                      exclude=[held_out], aug=aug_mode)
    val_ds = MachineFilteredDataset(root / "val", stage1_classes,
                                    exclude=[held_out])
    test_ds = MachineFilteredDataset(root / "test", stage1_classes,
                                     include=[held_out])

    if len(test_ds) < 10:
        print(f"  Only {len(test_ds)} test images from {held_out} — skipping")
        continue
    if len(val_ds) < 10:
        print(f"  Only {len(val_ds)} val images without {held_out} — skipping")
        continue

    for tag, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        cnt = np.bincount(ds.targets, minlength=len(stage1_classes))
        print(f"  {tag:5s} n={len(ds):5d} | " +
              "  ".join(f"{c}={n}" for c, n in zip(stage1_classes, cnt)))

    counts = np.bincount(train_ds.targets, minlength=len(stage1_classes)).astype(float)
    weights = (1.0 / np.maximum(counts, 1))[train_ds.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=WORKERS,
                          pin_memory=AMP, drop_last=True, persistent_workers=True)
    Xval, Yval = cache_images(val_ds)

    model = build_model(arch, len(stage1_classes), dropout)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best_f1, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        for x, y, _ in train_dl:
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                loss = criterion(model(to_input(x)), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
        pv = predict(model, Xval).argmax(1)
        vf1 = f1_score(Yval, pv, average="macro")
        vacc = accuracy_score(Yval, pv)
        flag = ""
        if vf1 > best_f1:
            best_f1, best_state, bad, flag = vf1, copy.deepcopy(model.state_dict()), 0, " <- best"
        else:
            bad += 1
        print(f"  ep {ep:02d} | loss {loss_sum / n_seen:.4f} | val acc {vacc:.4f} | "
              f"val F1 {vf1:.4f} | {time.time() - t0:.0f}s{flag}")
        if bad >= patience:
            print("  early stopping")
            break

    model.load_state_dict(best_state)
    Xtest, Ytest = cache_images(test_ds)
    P = predict(model, Xtest)
    pred = P.argmax(1)
    acc = float(accuracy_score(Ytest, pred))
    mf1 = float(f1_score(Ytest, pred, average="macro"))
    ece = ece_score(P, Ytest)
    bacc = float(balanced_accuracy_score(Ytest, pred))

    cm = confusion_matrix(Ytest, pred, labels=list(range(len(stage1_classes))))
    pd.DataFrame(cm, index=stage1_classes, columns=stage1_classes).to_csv(
        OUT_LMO / f"cm_without_{held_out.replace(' ', '_')}.csv")

    drop = baseline_acc - acc
    print(f"\n  RESULT on {held_out}: acc={acc:.4f} | macroF1={mf1:.4f} | "
          f"bal_acc={bacc:.4f} | ECE={ece:.4f} | drop from baseline={drop:+.4f}")
    results_lmo.append(dict(held_out_machine=held_out, n_train=len(train_ds),
                            n_test=len(test_ds), test_acc=round(acc, 4),
                            test_macroF1=round(mf1, 4), test_balanced_acc=round(bacc, 4),
                            test_ece=round(ece, 4), acc_drop=round(drop, 4)))

    torch.save({"model": best_state, "classes": stage1_classes,
                "held_out": held_out, "best_val_f1": best_f1},
               OUT_LMO / f"best_without_{held_out.replace(' ', '_')}.pth")
    del model, Xval, Xtest
    torch.cuda.empty_cache()

# ── summary ──
df_lmo = pd.DataFrame(results_lmo)
df_lmo.to_csv(OUT_LMO / "leave_machine_out_results.csv", index=False)
print("\n" + df_lmo.to_string(index=False))

# ── bar chart ──
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results_lmo))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(results_lmo)]
bars = ax.bar(x, [r["test_acc"] for r in results_lmo], color=colors, edgecolor="white", linewidth=1.2)
for i, r in enumerate(results_lmo):
    ax.text(i, r["test_acc"] + 0.015, f"{r['test_acc']:.3f}\n({r['acc_drop']:+.3f})",
            ha="center", fontsize=9, fontweight="bold")
ax.axhline(y=baseline_acc, color="gray", linestyle="--", alpha=0.7,
           label=f"All-machines baseline ({baseline_acc:.3f})")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['held_out_machine']}\n(n={r['n_test']})" for r in results_lmo],
                   fontsize=9)
ax.set_ylabel("Test Accuracy on Held-Out Machine")
ax.set_ylim(0, min(1.05, max(r["test_acc"] for r in results_lmo) + 0.12))
ax.set_title("Leave-Machine-Out: Does the Model Generalize Across Hardware?")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_LMO / "leave_machine_out.png", dpi=150, bbox_inches="tight")
plt.show()

# ── zip ──
import zipfile
zip_path = "/kaggle/working/LEAVE_MACHINE_OUT.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT_LMO.rglob("*")):
        if f.is_file() and not str(f).endswith(".pth"):
            zf.write(f, f.relative_to("/kaggle/working"))
print(f"\nZip: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
from IPython.display import FileLink
FileLink("LEAVE_MACHINE_OUT.zip")
