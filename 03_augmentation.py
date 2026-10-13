# 03_augmentation.py -- run every session (prints 'augmentation functions ready')
# Paste this whole file into ONE Colab cell and run it.
# Defines letterbox, fan_mask, load_image, augment. Needed by preview and training cells.

import cv2
import numpy as np

AUG = {
    "hflip_p": 0.5,
    "affine_p": 0.8, "rot_deg": 10, "scale": (0.9, 1.1), "shift": 0.06,   # probe position / zoom
    "speckle_p": 0.3, "looks": (10, 25),        # speckle (mild: images already NLM-denoised)
    "tgc_p": 0.5, "tgc_lin": 0.3, "tgc_quad": 0.2,   # depth-dependent gain (operator TGC)
    "gamma_p": 0.5, "gamma": (0.8, 1.25),       # machine dynamic range
    "degrade_p": 0.3, "degrade": (0.5, 0.85),   # lower-resolution machine
}

def letterbox(img, size):
    """Resize keeping aspect ratio, pad to a square with black (no stretching of anatomy)."""
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.zeros((size, size), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def fan_mask(img):
    """1 inside the ultrasound scan fan, 0 outside."""
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
    """Used for train, val AND test: grayscale -> letterbox -> black out everything outside the fan."""
    img = letterbox(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), size)
    return img * fan_mask(img)

def augment(img, mode="us", rng=None):
    """mode='us' = ultrasound-aware (ours), 'standard' = original-paper style (ablation)."""
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
    return np.clip(x * mask, 0, 255).astype(np.uint8)   # outside the fan stays pure black -> no streaks

print("augmentation functions ready")
