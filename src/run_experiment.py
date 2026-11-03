# Step 3: The main experiment (data collection).
# The power analysis said we need at least 11 runs per group. Runs only take a
# second each, so we do 20 per group to have some margin in case the pilot
# underestimated how much accuracy varies.
#
# Randomization: we make a list of 20 "clean" and 20 "noisy" runs, shuffle the
# order, and give every run its own random seed. The seed controls which images
# go into the training vs test set (and the noise, for noisy runs).

import random
import pandas as pd
from experiment import run_once, NOISE_SD

RUNS_PER_GROUP = 20

random.seed(42)
conditions = ["clean"] * RUNS_PER_GROUP + ["noisy"] * RUNS_PER_GROUP
random.shuffle(conditions)

# Seeds 1-40, one per run (the pilot used 1000+, so there's no overlap)
seeds = random.sample(range(1, 41), len(conditions))

rows = []
for run_order, (condition, seed) in enumerate(zip(conditions, seeds), start=1):
    acc = run_once(seed, condition)
    rows.append({"run_order": run_order, "condition": condition, "seed": seed,
                 "noise_sd": NOISE_SD if condition == "noisy" else 0.0,
                 "accuracy": acc})
    print(f"Run {run_order:2d} | {condition:5s} | seed {seed:2d} | accuracy = {acc:.4f}")

df = pd.DataFrame(rows)
df.to_csv("data/raw/experiment_results.csv", index=False)
print("\nSaved data/raw/experiment_results.csv")
