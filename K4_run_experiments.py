# K4_run_experiments.py -- Kaggle notebook
# Paste this whole file into ONE Kaggle cell. Needs K1, K2, K3 run first.
#
# STEP A: run it as it is (SMOKE = True). Takes ~5 min, checks everything works end to end.
#         Send Claude the output.
# STEP B: set SMOKE = False and run again. Trains the 4 Stage-1 runs one after another.
#         Tip: use "Save Version" -> "Save & Run All (Commit)" so it keeps running
#         even if you close the browser. Results land in /kaggle/working/runs/.
# STEP C (later): Stage 2 with the winning architecture -- uncomment the last line in RUNS.

SMOKE = True

RUNS = [
    dict(stage=1, arch="efficientnet_b0", aug="us"),
    dict(stage=1, arch="resnet50",        aug="us"),
    dict(stage=1, arch="densenet121",     aug="us"),
    dict(stage=1, arch="efficientnet_b0", aug="standard"),   # ablation: original-paper augmentation
    # dict(stage=2, arch="efficientnet_b0", aug="us"),       # Stage 2 -- set arch to the Stage-1 winner
]

if SMOKE:
    RUNS = RUNS[:1]

all_results = []
for cfg in RUNS:
    all_results.append(run_experiment(**cfg, smoke=SMOKE))
print("\nALL RUNS FINISHED")
