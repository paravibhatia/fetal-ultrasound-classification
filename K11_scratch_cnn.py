# K11_scratch_cnn.py — paste after K1, K2, K3 in a SEPARATE Kaggle notebook.
# Ablation: train a CNN from scratch (NO ImageNet pretraining).
# IT FACTOR: quantifies EXACTLY how much transfer learning contributes.
# If scratch CNN is much worse → ImageNet features are critical for ultrasound.
# If scratch CNN is close → our preprocessing (fan masking, US augmentation) is the real driver.
# Either outcome is a publishable finding. Most papers just say "we used pretrained" — we measure it.

import torch.nn as nn


class ScratchCNN(nn.Module):
    """5-block CNN trained from random init. ~3.5M params (vs 23-25M pretrained)."""
    def __init__(self, n_classes, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(512, n_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


n_params = sum(p.numel() for p in ScratchCNN(6).parameters()) / 1e6
print(f"ScratchCNN: {n_params:.1f}M params")

# monkey-patch build_model so run_experiment works with arch="scratch_cnn"
_orig_build = build_model

def build_model(arch, n_classes, p_drop):
    if arch == "scratch_cnn":
        return ScratchCNN(n_classes, p_drop).to(DEVICE)
    return _orig_build(arch, n_classes, p_drop)

# Stage 1, two runs: with US augmentation and without (same comparison as pretrained models)
# Higher lr (1e-3) and more epochs (50) since training from scratch converges slower.
run_experiment(stage=1, arch="scratch_cnn", aug="us",  epochs=50, patience=10,
               lr=1e-3, seed=42)
run_experiment(stage=1, arch="scratch_cnn", aug=None,  epochs=50, patience=10,
               lr=1e-3, seed=42)

print("\n" + "=" * 70)
print("DONE — compare these numbers to the pretrained ResNet-50 / DenseNet / EfficientNet results.")
print("The gap = how much transfer learning contributes.")
print("=" * 70)
