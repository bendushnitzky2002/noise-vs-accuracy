# Step 2: Power analysis to figure out how many runs we need per group.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

pilot = pd.read_csv("data/raw/pilot_results.csv")
pilot["accuracy_pct"] = pilot["accuracy"] * 100

clean = pilot[pilot["condition"] == "clean"]["accuracy_pct"]
noisy = pilot[pilot["condition"] == "noisy"]["accuracy_pct"]

# Pooled standard deviation (same formula as in class)
n1 = len(clean)
n2 = len(noisy)
pooled_sd = np.sqrt(((n1 - 1) * clean.std()**2 + (n2 - 1) * noisy.std()**2) / (n1 + n2 - 2))

# The smallest drop in accuracy I care about is 1 percentage point
# (about 1 extra mistake for every 100 digits)
min_effect = 1.0
effect_size = min_effect / pooled_sd

print("Pilot SD (clean):", round(clean.std(), 3))
print("Pilot SD (noisy):", round(noisy.std(), 3))
print("Pooled SD:", round(pooled_sd, 3))
print("Cohen's d:", round(effect_size, 2))

analysis = TTestIndPower()
n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
n_needed = int(np.ceil(n_needed))
print("Required sample size:", n_needed, "runs per group")

with open("results/power_analysis.txt", "w") as f:
    f.write(f"Pilot SD (clean): {clean.std():.3f}\n")
    f.write(f"Pilot SD (noisy): {noisy.std():.3f}\n")
    f.write(f"Pooled SD: {pooled_sd:.3f}\n")
    f.write(f"Smallest effect we care about: {min_effect} percentage point\n")
    f.write(f"Cohen's d: {effect_size:.2f}\n")
    f.write("alpha = 0.05, power = 0.80, two-sided\n")
    f.write(f"Required sample size: {n_needed} runs per group\n")

# Plot how power changes with the number of runs per group
ns = np.arange(2, 41)
powers = analysis.power(effect_size=effect_size, nobs1=ns, alpha=0.05)

plt.figure(figsize=(7, 4))
plt.plot(ns, powers, color="#2a78d6", linewidth=2)
plt.axhline(0.80, color="gray", linestyle="--")
plt.axvline(n_needed, color="gray", linestyle=":")
plt.xlabel("Runs per group")
plt.ylabel("Power")
plt.title("Power to detect a 1-point drop in accuracy")
plt.tight_layout()
plt.savefig("figures/power_curve.png", dpi=150)
print("Saved results/power_analysis.txt and figures/power_curve.png")
