# Step 1: Small pilot study.
# Runs 10 models per condition to see how much accuracy varies from run to run.
# That spread is what the power analysis needs. The pilot uses different seeds
# (1000+) than the main experiment so the pilot data isn't reused later.

import pandas as pd
from experiment import run_once

PILOT_RUNS = 10

rows = []
for condition in ["clean", "noisy"]:
    for i in range(PILOT_RUNS):
        seed = 1000 + i if condition == "clean" else 1100 + i
        acc = run_once(seed, condition)
        rows.append({"condition": condition, "seed": seed, "accuracy": acc})
        print(f"{condition} run {i + 1}: accuracy = {acc:.4f}")

df = pd.DataFrame(rows)
df.to_csv("data/raw/pilot_results.csv", index=False)
print("\nSaved data/raw/pilot_results.csv")
print(df.groupby("condition")["accuracy"].agg(["mean", "std"]))
