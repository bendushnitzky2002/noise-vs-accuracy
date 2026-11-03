# Step 4: Check and clean the raw results.

import pandas as pd

df = pd.read_csv("data/raw/experiment_results.csv")
print("Raw rows:", len(df))

# Basic checks
print("Missing values:\n", df.isna().sum())
print("Duplicate seeds:", df["seed"].duplicated().sum())
print("Runs per group:\n", df["condition"].value_counts())

# Accuracy has to be between 0 and 1. Anything else would be a bug.
bad = df[(df["accuracy"] < 0) | (df["accuracy"] > 1)]
print("Impossible accuracy values:", len(bad))

df = df.dropna().drop_duplicates(subset="seed")
df = df[(df["accuracy"] >= 0) & (df["accuracy"] <= 1)]

# Accuracy as a percentage is easier to read
df["accuracy_pct"] = df["accuracy"] * 100

df.to_csv("data/clean/experiment_results_clean.csv", index=False)
print("\nClean rows:", len(df))
print("Saved data/clean/experiment_results_clean.csv")
