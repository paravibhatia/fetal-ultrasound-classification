# K6_save_everything.py — paste into the LAST cell, run AFTER all 10 runs finish
# Creates one zip with all figures, CSVs, JSONs (skips .pth model weights to avoid hanging)

import zipfile, glob, os, json
import pandas as pd

# ── results table ──
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

# ── zip everything except .pth ──
zip_path = "/kaggle/working/FINAL_RESULTS.zip"
count = 0
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(glob.glob("/kaggle/working/runs/**/*", recursive=True)):
        if os.path.isfile(f) and not f.endswith(".pth"):
            zf.write(f, os.path.relpath(f, "/kaggle/working"))
            count += 1

size_mb = round(os.path.getsize(zip_path) / 1e6, 1)
print(f"\nFINAL_RESULTS.zip — {count} files, {size_mb} MB")

from IPython.display import FileLink
FileLink("FINAL_RESULTS.zip")
