# K5_results_table.py -- Kaggle notebook
# Paste this whole file into ONE Kaggle cell. Shows every finished run side by side.

import json, glob
import pandas as pd

rows = []
for f in sorted(glob.glob(str(OUT / "*" / "results.json"))):
    r = json.load(open(f))
    row = {"run": r["run"], "val_macroF1": r["best_val_macro_f1"], "test_acc": r["test_acc"],
           "test_macroF1": r["test_macro_f1"], "balanced_acc": r["test_balanced_acc"],
           "ECE": r["test_ece"], "MC_acc": r["mc_test_acc"], "TTA_acc": r.get("tta_test_acc"), "TTA_macroF1": r.get("tta_test_macro_f1"), "ms/img": r["inference_ms_per_img"],
           "acc@80%": r["risk_coverage"]["acc@80%"]}
    for m, v in r["acc_by_machine"].items():
        row[f"acc_{m}"] = v["acc"]
    rows.append(row)

table = pd.DataFrame(rows).round(4)
table.to_csv(OUT / "results_table.csv", index=False)
table
