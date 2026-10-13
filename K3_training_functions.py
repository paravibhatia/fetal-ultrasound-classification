# K3_training_functions.py -- Kaggle notebook, run every session (prints 'training functions ready')
# Paste this whole file into ONE Kaggle cell. Needs K1 and K2 run first.
#
# Defines run_experiment(...). Nothing trains until you run K4.
#
# What makes this more than a basic training script (write these in Methods):
#   1. Patient-wise train / val / test. Model selection uses val only; test is touched once.
#   2. Every image: letterbox (no stretching) + fan-ROI mask (outside the scan fan = black).
#   3. Ultrasound-aware augmentation applied on the fly (new random version each epoch).
#   4. Class balance via a class-balanced sampler (no class-weighted loss -> no double correction).
#   5. Dropout head -> Monte Carlo Dropout uncertainty at test time.
#   6. Best epoch chosen by val MACRO-F1 (accuracy hides the rare classes).
#   7. Saved for the paper: accuracy, macro-F1, balanced accuracy, ECE (calibration),
#      MC-Dropout entropy + mutual information, risk-coverage table, per-machine accuracy,
#      confusion matrix, per-image predictions CSV (used later for Grad-CAM / error analysis).
#   8. Test-time augmentation (TTA): original + flip + 4 small shifts, averaged. Reported next to
#      plain results, and TTA disagreement is saved as a second uncertainty signal next to MC Dropout.

import time, json, copy, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)

cv2.setNumThreads(0)                      # avoid CPU oversubscription inside DataLoader workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
WORKERS = 4
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

META = pd.read_excel(DATA / "FETAL_PLANES_DB_data.xlsx")
META.columns = [c.strip() for c in META.columns]
MACHINE_OF = dict(zip(META["Image_name"].astype(str).str.strip(), META["US_Machine"]))


# ---------------- data ----------------
class USFolder(Dataset):
    def __init__(self, root, classes=None, aug=None, size=224):
        root = Path(root)
        self.classes = classes or sorted(d.name for d in root.iterdir() if d.is_dir())
        self.samples = [(str(f), i) for i, c in enumerate(self.classes) if (root / c).exists()
                        for f in sorted((root / c).glob("*.png"))]
        self.targets = [t for _, t in self.samples]
        self.aug, self.size = aug, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = load_image(path, self.size)                       # letterbox + fan mask (K2)
        if self.aug:
            img = augment(img, self.aug, np.random.default_rng())  # K2
        return torch.from_numpy(img), y, i


def to_input(x_uint8):
    """uint8 (B,H,W) on CPU -> normalized 3-channel float on GPU."""
    x = x_uint8.to(DEVICE, non_blocking=True).unsqueeze(1).float().div(255).expand(-1, 3, -1, -1)
    return (x - MEAN) / STD


def cache_images(ds):
    """Load an eval set once into RAM (uint8) so val/test/MC passes are fast."""
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=WORKERS)
    X, Y = [], []
    for x, y, _ in dl:
        X.append(x); Y.append(y)
    return torch.cat(X), torch.cat(Y).numpy()


# ---------------- model ----------------
def build_model(arch, n_classes, p_drop):
    if arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Sequential(nn.Dropout(p_drop), nn.Linear(m.fc.in_features, n_classes))
    elif arch == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        m.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(m.classifier.in_features, n_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(m.classifier[1].in_features, n_classes))
    else:
        raise ValueError(arch)
    return m.to(DEVICE)


@torch.no_grad()
def predict(model, X, mc_T=0, bs=128):
    """Returns softmax probs. mc_T>0 -> Monte Carlo Dropout (T passes, dropout on, BatchNorm frozen)."""
    model.eval()
    if mc_T:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()
    runs = []
    for _ in range(max(1, mc_T)):
        P = []
        for i in range(0, len(X), bs):
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                P.append(torch.softmax(model(to_input(X[i:i + bs])).float(), 1).cpu())
        runs.append(torch.cat(P).numpy())
    model.eval()
    return np.stack(runs) if mc_T else runs[0]


def tta_views(x, shift=8):
    """Test-time augmentation views of a uint8 batch (B,H,W): original, flip, 4 small shifts (black fill).
    Only MILD, realistic changes -- no speckle/blur at test time."""
    import torch.nn.functional as F
    views = [x, torch.flip(x, dims=[2])]
    p = F.pad(x, (shift, shift, shift, shift))
    H, W = x.shape[1:]
    for dy, dx in [(0, -shift), (0, shift), (-shift, 0), (shift, 0)]:
        views.append(p[:, shift + dy:shift + dy + H, shift + dx:shift + dx + W])
    return views


@torch.no_grad()
def predict_tta(model, X, bs=128):
    """Returns (V x N x C) softmax probs, one slice per TTA view."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        batch = []
        for v in tta_views(X[i:i + bs]):
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                batch.append(torch.softmax(model(to_input(v)).float(), 1).cpu())
        out.append(torch.stack(batch))
    return torch.cat(out, dim=1).numpy()


# ---------------- metrics ----------------
def ece_score(probs, labels, n_bins=15):
    """Expected Calibration Error: gap between confidence and actual accuracy."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage(uncertainty, correct, coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Auto-accept the most certain X% of images, send the rest to a clinician."""
    order = np.argsort(uncertainty)
    return {f"acc@{int(c * 100)}%": float(correct[order[:max(1, round(c * len(order)))]].mean()) for c in coverages}


# ---------------- experiment ----------------
def run_experiment(stage, arch, aug="us", epochs=25, patience=6, bs=32, lr=2e-4, wd=1e-4,
                   dropout=0.3, mc_T=20, seed=42, smoke=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = DATA if stage == 1 else DATA / "brain_subplanes"
    name = f"stage{stage}_{arch}_{aug}_seed{seed}" + ("_SMOKE" if smoke else "")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(stage=stage, arch=arch, aug=aug, epochs=epochs, patience=patience, bs=bs, lr=lr,
               wd=wd, dropout=dropout, mc_T=mc_T, seed=seed, smoke=smoke)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"\n{'=' * 70}\nRUN: {name}\n{'=' * 70}")

    train_ds = USFolder(root / "train", aug=aug)
    classes = train_ds.classes
    val_ds = USFolder(root / "val", classes=classes)
    test_ds = USFolder(root / "test", classes=classes)
    if smoke:   # tiny random subsets, just to check everything runs end to end
        epochs, mc_T = 1, 2
        for ds in (train_ds, val_ds, test_ds):
            ds.samples = random.sample(ds.samples, min(300, len(ds.samples)))
            ds.targets = [t for _, t in ds.samples]
    for tag, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        cnt = np.bincount(ds.targets, minlength=len(classes))
        print(f"{tag:5s} n={len(ds):5d} | " + "  ".join(f"{c}={n}" for c, n in zip(classes, cnt)))

    counts = np.bincount(train_ds.targets, minlength=len(classes)).astype(float)
    weights = (1.0 / np.maximum(counts, 1))[train_ds.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=WORKERS,
                          pin_memory=AMP, drop_last=True, persistent_workers=True)
    Xval, Yval = cache_images(val_ds)

    model = build_model(arch, len(classes), dropout)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best_f1, best_state, bad, history = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); loss_sum, n_seen = 0.0, 0
        for x, y, _ in train_dl:
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                loss = criterion(model(to_input(x)), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * len(y); n_seen += len(y)
        pv = predict(model, Xval).argmax(1)
        vf1, vacc = f1_score(Yval, pv, average="macro"), accuracy_score(Yval, pv)
        history.append(dict(epoch=ep, train_loss=loss_sum / n_seen, val_acc=vacc, val_macro_f1=vf1))
        flag = ""
        if vf1 > best_f1:
            best_f1, best_state, bad, flag = vf1, copy.deepcopy(model.state_dict()), 0, "  <- best"
        else:
            bad += 1
        print(f"ep {ep:02d} | loss {loss_sum / n_seen:.4f} | val acc {vacc:.4f} | val macroF1 {vf1:.4f} | {time.time() - t0:.0f}s{flag}")
        if bad >= patience:
            print("early stopping"); break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "classes": classes, **cfg}, out / "best.pth")

    # ---------- TEST (touched once) ----------
    Xtest, Ytest = cache_images(test_ds)
    if AMP: torch.cuda.synchronize()
    t0 = time.time()
    P = predict(model, Xtest)
    if AMP: torch.cuda.synchronize()
    ms_per_img = 1000 * (time.time() - t0) / len(Ytest)
    pred = P.argmax(1)

    runs = predict(model, Xtest, mc_T=mc_T)                     # T x N x C
    Pmc = runs.mean(0)
    entropy = -(Pmc * np.log(Pmc + 1e-12)).sum(1)
    mutual_info = entropy - (-(runs * np.log(runs + 1e-12)).sum(2)).mean(0)
    mc_pred = Pmc.argmax(1)
    correct_mc = (mc_pred == Ytest).astype(float)

    tta_runs = predict_tta(model, Xtest)                         # V x N x C
    Ptta = tta_runs.mean(0)
    tta_pred = Ptta.argmax(1)
    tta_entropy = -(Ptta * np.log(Ptta + 1e-12)).sum(1)
    tta_disagree = (tta_runs.argmax(2) != tta_pred[None]).mean(0)   # share of views that disagree

    names = [Path(test_ds.samples[i][0]).stem for i in range(len(test_ds))]
    df = pd.DataFrame({"image": names, "machine": [MACHINE_OF.get(n, "?") for n in names],
                       "true": [classes[t] for t in Ytest], "pred": [classes[p] for p in pred],
                       "conf": P.max(1), "mc_pred": [classes[p] for p in mc_pred], "mc_conf": Pmc.max(1),
                       "mc_entropy": entropy, "mc_mutual_info": mutual_info,
                       "tta_pred": [classes[p] for p in tta_pred], "tta_conf": Ptta.max(1),
                       "tta_entropy": tta_entropy, "tta_disagreement": tta_disagree})
    for ci, c in enumerate(classes):
        df[f"p_{c}"] = P[:, ci]
    df.to_csv(out / "test_predictions.csv", index=False)

    acc_by_machine = df.assign(ok=df.true == df.pred).groupby("machine")["ok"].agg(["mean", "size"])
    results = dict(run=name, best_val_macro_f1=float(best_f1), epochs_run=len(history),
                   test_acc=float(accuracy_score(Ytest, pred)),
                   test_macro_f1=float(f1_score(Ytest, pred, average="macro")),
                   test_balanced_acc=float(balanced_accuracy_score(Ytest, pred)),
                   test_ece=ece_score(P, Ytest),
                   mc_test_acc=float(correct_mc.mean()), mc_test_ece=ece_score(Pmc, Ytest),
                   tta_test_acc=float(accuracy_score(Ytest, tta_pred)),
                   tta_test_macro_f1=float(f1_score(Ytest, tta_pred, average="macro")),
                   tta_test_ece=ece_score(Ptta, Ytest),
                   risk_coverage_tta=risk_coverage(tta_entropy, (tta_pred == Ytest).astype(float)),
                   inference_ms_per_img=ms_per_img,
                   risk_coverage=risk_coverage(entropy, correct_mc),
                   acc_by_machine={m: dict(acc=float(r["mean"]), n=int(r["size"])) for m, r in acc_by_machine.iterrows()})
    json.dump(results, open(out / "results.json", "w"), indent=2)
    report = classification_report(Ytest, pred, target_names=classes, digits=4)
    (out / "classification_report.txt").write_text(report)
    pd.DataFrame(confusion_matrix(Ytest, pred), index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")

    print("\n" + report)
    print(f"test acc {results['test_acc']:.4f} | macroF1 {results['test_macro_f1']:.4f} | "
          f"balanced acc {results['test_balanced_acc']:.4f} | ECE {results['test_ece']:.4f} | {ms_per_img:.2f} ms/img")
    print(f"with TTA: acc {results['tta_test_acc']:.4f} | macroF1 {results['tta_test_macro_f1']:.4f} | ECE {results['tta_test_ece']:.4f}")
    print("risk-coverage (MC Dropout):", {k: round(v, 4) for k, v in results["risk_coverage"].items()})
    print("accuracy by machine:", {m: round(v["acc"], 4) for m, v in results["acc_by_machine"].items()})
    print("saved to", out)
    return results


print("training functions ready | device:", DEVICE)
