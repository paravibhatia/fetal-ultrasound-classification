# =====================================================================
# FULL NOTEBOOK — paste each section into a SEPARATE Kaggle cell
# Sections are marked with: # ==== CELL 1 ====, # ==== CELL 2 ==== etc.
# Run in order. Then do Save Version -> Save & Run All (Commit).
# =====================================================================


# ==== CELL 1: SETUP ====
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
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")


# ==== CELL 2: AUGMENTATION FUNCTIONS ====
import cv2
import numpy as np

AUG = {
    "hflip_p": 0.5,
    "affine_p": 0.8, "rot_deg": 10, "scale": (0.9, 1.1), "shift": 0.06,
    "speckle_p": 0.3, "looks": (10, 25),
    "tgc_p": 0.5, "tgc_lin": 0.3, "tgc_quad": 0.2,
    "gamma_p": 0.5, "gamma": (0.8, 1.25),
    "degrade_p": 0.3, "degrade": (0.5, 0.85),
}

def letterbox(img, size):
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.zeros((size, size), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def fan_mask(img):
    m = (cv2.medianBlur(img, 5) > 15).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(img)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pts = np.column_stack(np.where(lab == k))[:, ::-1].astype(np.int32)
    out = np.zeros_like(img)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out if 0.15 < out.mean() < 0.97 else np.ones_like(img)

def load_image(path, size=224):
    img = letterbox(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), size)
    return img * fan_mask(img)

def augment(img, mode="us", rng=None):
    rng = rng or np.random.default_rng()
    p, mask, x = AUG, fan_mask(img), img
    h, w = img.shape
    if rng.random() < p["hflip_p"]:
        x, mask = cv2.flip(x, 1), cv2.flip(mask, 1)
    if rng.random() < p["affine_p"]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-p["rot_deg"], p["rot_deg"]), rng.uniform(*p["scale"]))
        M[:, 2] += rng.uniform(-p["shift"], p["shift"], 2) * [w, h]
        x = cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = x.astype(np.float32)
    if mode == "standard":
        x = x * (1 + rng.uniform(-0.2, 0.2)) + rng.uniform(-0.2, 0.2) * 128
    else:
        if rng.random() < p["speckle_p"]:
            L = rng.uniform(*p["looks"])
            noise = cv2.GaussianBlur(rng.gamma(L, 1 / L, x.shape).astype(np.float32), (0, 0), 0.8)
            x = x * noise / noise.mean()
        if rng.random() < p["tgc_p"]:
            rows = np.where(mask.any(1))[0]
            if len(rows) > 1:
                d = np.clip((np.arange(h) - rows[0]) / max(rows[-1] - rows[0], 1), 0, 1)
                gain = 1 + rng.uniform(-p["tgc_lin"], p["tgc_lin"]) * d + rng.uniform(-p["tgc_quad"], p["tgc_quad"]) * d ** 2
                x = x * np.clip(gain, 0.6, 1.4)[:, None]
        if rng.random() < p["gamma_p"]:
            x = 255 * (np.clip(x, 0, 255) / 255) ** rng.uniform(*p["gamma"])
        if rng.random() < p["degrade_p"]:
            s = rng.uniform(*p["degrade"])
            x = cv2.resize(cv2.resize(x, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    return np.clip(x * mask, 0, 255).astype(np.uint8)

print("augmentation functions ready")


# ==== CELL 3: TRAINING FUNCTIONS ====
import time, json, copy, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)

cv2.setNumThreads(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
WORKERS = 4
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

META = pd.read_excel(DATA / "FETAL_PLANES_DB_data.xlsx")
META.columns = [c.strip() for c in META.columns]
MACHINE_OF = dict(zip(META["Image_name"].astype(str).str.strip(), META["US_Machine"]))

class USFolder(Dataset):
    def __init__(self, root, classes=None, aug=None, size=224):
        root = Path(root)
        self.classes = classes or sorted(d.name for d in root.iterdir() if d.is_dir())
        self.samples = [(str(f), i) for i, c in enumerate(self.classes) if (root / c).exists()
                        for f in sorted((root / c).glob("*.png"))]
        self.targets = [t for _, t in self.samples]
        self.aug, self.size = aug, size
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = load_image(path, self.size)
        if self.aug:
            img = augment(img, self.aug, np.random.default_rng())
        return torch.from_numpy(img), y, i

def to_input(x_uint8):
    x = x_uint8.to(DEVICE, non_blocking=True).unsqueeze(1).float().div(255).expand(-1, 3, -1, -1)
    return (x - MEAN) / STD

def cache_images(ds):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=WORKERS)
    X, Y = [], []
    for x, y, _ in dl:
        X.append(x); Y.append(y)
    return torch.cat(X), torch.cat(Y).numpy()

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
    model.eval()
    if mc_T:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout): mod.train()
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
    import torch.nn.functional as F
    views = [x, torch.flip(x, dims=[2])]
    p = F.pad(x, (shift, shift, shift, shift))
    H, W = x.shape[1:]
    for dy, dx in [(0, -shift), (0, shift), (-shift, 0), (shift, 0)]:
        views.append(p[:, shift + dy:shift + dy + H, shift + dx:shift + dx + W])
    return views

@torch.no_grad()
def predict_tta(model, X, bs=128):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        batch = []
        for v in tta_views(X[i:i + bs]):
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                batch.append(torch.softmax(model(to_input(v)).float(), 1).cpu())
        out.append(torch.stack(batch))
    return torch.cat(out, dim=1).numpy()

def ece_score(probs, labels, n_bins=15):
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
    order = np.argsort(uncertainty)
    return {f"acc@{int(c * 100)}%": float(correct[order[:max(1, round(c * len(order)))]].mean()) for c in coverages}

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
    if smoke:
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
    Xtest, Ytest = cache_images(test_ds)
    if AMP: torch.cuda.synchronize()
    t0 = time.time()
    P = predict(model, Xtest)
    if AMP: torch.cuda.synchronize()
    ms_per_img = 1000 * (time.time() - t0) / len(Ytest)
    pred = P.argmax(1)
    runs = predict(model, Xtest, mc_T=mc_T)
    Pmc = runs.mean(0)
    entropy = -(Pmc * np.log(Pmc + 1e-12)).sum(1)
    mutual_info = entropy - (-(runs * np.log(runs + 1e-12)).sum(2)).mean(0)
    mc_pred = Pmc.argmax(1)
    correct_mc = (mc_pred == Ytest).astype(float)
    tta_runs = predict_tta(model, Xtest)
    Ptta = tta_runs.mean(0)
    tta_pred = Ptta.argmax(1)
    tta_entropy = -(Ptta * np.log(Ptta + 1e-12)).sum(1)
    tta_disagree = (tta_runs.argmax(2) != tta_pred[None]).mean(0)
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


# ==== CELL 4: ALL 10 RUNS + PER-RUN FIGURES ====
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.use("Agg")
mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def figs_for(res):
    d = OUT / res["run"]
    h = pd.read_csv(d / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    ax[0].plot(h.epoch, h.train_loss, color="#eb6834", lw=1.5)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss (log)")
    ax[0].set_title("Training loss", loc="left")
    ax[1].plot(h.epoch, h.val_macro_f1, color="#2a78d6", lw=1.5)
    ax[1].axhline(h.val_macro_f1.max(), ls="--", lw=1, color="#898781")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val macro-F1")
    ax[1].set_title(f"Validation macro-F1  (best {h.val_macro_f1.max():.4f})", loc="left")
    fig.suptitle(res["run"], fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(d / "fig_curves.png"); plt.close(fig)
    cm = pd.read_csv(d / "confusion_matrix.csv", index_col=0)
    cmn = cm.div(cm.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    lbl = [c.replace("_", "\n") for c in cm.columns]
    ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=7)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cmn.iloc[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "#14181B")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(res["run"], loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(d / "fig_confusion.png"); plt.close(fig)
    df = pd.read_csv(d / "test_predictions.csv")
    ok = (df.true == df.pred).astype(float).values; conf = df.conf.values
    edges = np.linspace(0, 1, 16); mid, acc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 20:
            mid.append((lo + hi) / 2); acc.append(ok[m].mean())
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="#898781", lw=1, label="perfect calibration")
    ax.bar(mid, acc, width=1/15*0.9, color="#2a78d6", edgecolor="white", lw=0.6, label="observed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("model confidence"); ax.set_ylabel("actual accuracy")
    ax.set_title(f"Reliability — ECE {res['test_ece']:.4f}", loc="left", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(d / "fig_reliability.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for tag, key, c, mk in [("MC Dropout", "risk_coverage", "#2a78d6", "o"),
                            ("TTA disagreement", "risk_coverage_tta", "#eb6834", "s")]:
        rc = res.get(key)
        if not rc: continue
        cov = np.array([int(k.split("@")[1].strip("%")) for k in rc]); val = np.array(list(rc.values()))
        o = np.argsort(cov)
        ax.plot(cov[o], val[o], marker=mk, color=c, lw=1.6, ms=4, label=tag)
    ax.axhline(res["test_acc"], ls="--", lw=1, color="#898781")
    ax.set_xlabel("coverage — % auto-accepted"); ax.set_ylabel("accuracy on accepted")
    ax.set_title("Selective prediction", loc="left", fontsize=9)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(d / "fig_risk_coverage.png"); plt.close(fig)
    print(f"   figures written to {d.name}")

SMOKE = False
RUNS = [
    dict(stage=1, arch="efficientnet_b0", aug=None,       epochs=50, patience=15),
    dict(stage=1, arch="resnet50",        aug=None,       epochs=50, patience=15),
    dict(stage=1, arch="densenet121",     aug=None,       epochs=50, patience=15),
    dict(stage=1, arch="efficientnet_b0", aug="standard", epochs=50, patience=15),
    dict(stage=1, arch="efficientnet_b0", aug="us",       epochs=50, patience=15),
    dict(stage=1, arch="resnet50",        aug="us",       epochs=50, patience=15),
    dict(stage=1, arch="densenet121",     aug="us",       epochs=50, patience=15),
    dict(stage=2, arch="efficientnet_b0", aug=None,       epochs=50, patience=15),
    dict(stage=2, arch="efficientnet_b0", aug="us",       epochs=50, patience=15),
    dict(stage=2, arch="densenet121",     aug="us",       epochs=50, patience=15),
]

all_results = []
for i, cfg in enumerate(RUNS, 1):
    print(f"\n########## RUN {i} of {len(RUNS)} ##########")
    r = run_experiment(**cfg, smoke=SMOKE)
    figs_for(r)
    all_results.append(r)
print("\nALL RUNS FINISHED")


# ==== CELL 5: RESULTS TABLE ====
import json, glob
import pandas as pd

rows = []
for f in sorted(glob.glob(str(OUT / "*" / "results.json"))):
    r = json.load(open(f))
    row = {"run": r["run"], "val_macroF1": r["best_val_macro_f1"], "test_acc": r["test_acc"],
           "test_macroF1": r["test_macro_f1"], "balanced_acc": r["test_balanced_acc"],
           "ECE": r["test_ece"], "MC_acc": r["mc_test_acc"], "TTA_acc": r.get("tta_test_acc"),
           "TTA_macroF1": r.get("tta_test_macro_f1"), "ms/img": r["inference_ms_per_img"],
           "acc@80%": r["risk_coverage"]["acc@80%"]}
    for m, v in r["acc_by_machine"].items():
        row[f"acc_{m}"] = v["acc"]
    rows.append(row)

table = pd.DataFrame(rows).round(4)
table.to_csv(OUT / "results_table.csv", index=False)
print(table.to_string())
print(f"\n{len(rows)} runs found")


# ==== CELL 6: GRAD-CAM ON RESNET-50 US-AUG ====
import matplotlib.pyplot as plt

best_run = "stage1_resnet50_us_seed42"
ckpt = torch.load(OUT / best_run / "best.pth", map_location=DEVICE)
gcmodel = build_model("resnet50", len(ckpt["classes"]), 0.3)
gcmodel.load_state_dict(ckpt["model"])
gcmodel.eval()
gc_classes = ckpt["classes"]

def gradcam(model, x_uint8, target_class=None):
    x = x_uint8.unsqueeze(0).to(DEVICE).unsqueeze(1).float().div(255).expand(-1, 3, -1, -1)
    x = (x - MEAN) / STD
    x.requires_grad_(True)
    feats, grads = [], []
    h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = model.layer4.register_full_backward_hook(lambda m, i, o: grads.append(o[0]))
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(1).item()
    model.zero_grad()
    logits[0, target_class].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = (w * feats[0]).sum(1).squeeze().relu()
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                          size=(224, 224), mode="bilinear").squeeze()
    return cam.detach().cpu().numpy(), target_class

test_ds = USFolder(DATA / "test", classes=gc_classes)
preds_df = pd.read_csv(OUT / best_run / "test_predictions.csv")

samples = []
for cls in gc_classes:
    correct = preds_df[(preds_df.true == cls) & (preds_df.pred == cls)].head(3)
    wrong = preds_df[(preds_df.true == cls) & (preds_df.pred != cls)].head(2)
    for _, row in pd.concat([correct, wrong]).iterrows():
        samples.append({"image": row.image, "true": row.true, "pred": row.pred,
                        "correct": row.true == row.pred})

gc_dir = OUT / best_run / "gradcam"
gc_dir.mkdir(exist_ok=True)
fan_ratios = {cls: [] for cls in gc_classes}

for s in samples:
    path = None
    for cls_dir in (DATA / "test").iterdir():
        candidate = cls_dir / (s["image"] + ".png")
        if candidate.exists():
            path = str(candidate); break
    if path is None:
        continue
    img = load_image(path, 224)
    cam, pred_cls = gradcam(gcmodel, torch.from_numpy(img))
    fan = fan_mask(img).astype(np.float32)
    if fan.shape != (224, 224):
        fan = cv2.resize(fan, (224, 224))
    in_fan = float((cam * fan).sum() / (cam.sum() + 1e-8))
    fan_ratios[s["true"]].append(in_fan)
    img_rgb = np.stack([img, img, img], axis=-1)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_rgb.astype(np.float32) / 255.0 + 0.5 * heatmap.astype(np.float32) / 255.0)
    overlay = (overlay / overlay.max() * 255).astype(np.uint8)
    tag = "CORRECT" if s["correct"] else f"WRONG_pred_{s['pred']}"
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original", fontsize=9)
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title(f"Grad-CAM (in-fan: {in_fan:.2f})", fontsize=9)
    axes[2].imshow(overlay); axes[2].set_title(f"True: {s['true']} | Pred: {s['pred']}", fontsize=9)
    for a in axes: a.axis("off")
    fig.tight_layout()
    fig.savefig(gc_dir / f"{s['true']}_{s['image']}_{tag}.png", dpi=200)
    plt.close(fig)

print(f"\n{len(list(gc_dir.glob('*.png')))} Grad-CAM images saved")

# in-fan ratio bar chart
fig, ax = plt.subplots(figsize=(8, 4))
cls_names = [c for c in gc_classes if fan_ratios[c]]
means = [np.mean(fan_ratios[c]) for c in cls_names]
stds = [np.std(fan_ratios[c]) for c in cls_names]
colors = ["#2a78d6" if m > 0.7 else "#eb6834" if m > 0.5 else "#d62728" for m in means]
ax.bar(range(len(cls_names)), means, yerr=stds, color=colors, edgecolor="white", capsize=4)
ax.set_xticks(range(len(cls_names)))
ax.set_xticklabels([c.replace("_", "\n") for c in cls_names], fontsize=8)
ax.set_ylabel("In-Fan Attention Ratio")
ax.set_title("Grad-CAM: fraction of attention inside ultrasound fan", loc="left")
ax.axhline(0.8, ls="--", lw=1, color="#898781", label="80% threshold")
ax.set_ylim(0, 1.05); ax.legend(frameon=False)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.03, f'{m:.2f}', ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(gc_dir / "fig_infan_ratio.png", dpi=300)
plt.close(fig)

json.dump(fan_ratios, open(gc_dir / "fan_ratios.json", "w"), indent=2)
print("in-fan ratio chart saved")


# ==== CELL 7: ZIP EVERYTHING (INCLUDING .pth) ====
import zipfile, os

zip_path = "/kaggle/working/FINAL_RESULTS.zip"
count = 0
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(glob.glob("/kaggle/working/runs/**/*", recursive=True)):
        if os.path.isfile(f):
            zf.write(f, os.path.relpath(f, "/kaggle/working"))
            count += 1

size_mb = round(os.path.getsize(zip_path) / 1e6, 1)
print(f"\nFINAL_RESULTS.zip — {count} files, {size_mb} MB")
print("THIS ZIP INCLUDES .pth MODEL FILES")

from IPython.display import FileLink
FileLink("FINAL_RESULTS.zip")
