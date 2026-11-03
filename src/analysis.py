# Step 5: Statistical analysis and figures.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from experiment import load_images, add_noise, NOISE_SD

df = pd.read_csv("data/clean/experiment_results_clean.csv")
clean = df[df["condition"] == "clean"]["accuracy_pct"]
noisy = df[df["condition"] == "noisy"]["accuracy_pct"]

# ----- Descriptive statistics -----
print(df.groupby("condition")["accuracy_pct"].describe().round(2))

# ----- Check assumptions -----
# Shapiro-Wilk: H0 = the data is normal, so a small p-value would be a problem
sw_clean = stats.shapiro(clean)
sw_noisy = stats.shapiro(noisy)
print("\nShapiro-Wilk clean:", sw_clean)
print("Shapiro-Wilk noisy:", sw_noisy)

# Levene: H0 = equal variances
levene = stats.levene(clean, noisy)
print("Levene:", levene)

# ----- Welch's t-test -----
# Welch because it doesn't assume equal variances (equal_var=False)
t_test = stats.ttest_ind(clean, noisy, equal_var=False)
ci = t_test.confidence_interval(confidence_level=0.95)
diff = clean.mean() - noisy.mean()
print("\nWelch's t-test:", t_test)
print("Mean difference:", round(diff, 2), "percentage points")
print("95% CI:", round(ci.low, 2), "to", round(ci.high, 2))

# Cohen's d with the pooled SD (same formula as in class)
n1 = len(clean)
n2 = len(noisy)
pooled_sd = np.sqrt(((n1 - 1) * clean.std()**2 + (n2 - 1) * noisy.std()**2) / (n1 + n2 - 2))
d = diff / pooled_sd
print("Cohen's d:", round(d, 2))

# ----- Mann-Whitney U as a backup (doesn't assume normality) -----
mw = stats.mannwhitneyu(clean, noisy)
print("Mann-Whitney U:", mw)

# Save the main numbers to a text file
with open("results/stats_summary.txt", "w") as f:
    f.write(f"Clean: mean = {clean.mean():.2f}%, SD = {clean.std():.2f}, n = {n1}\n")
    f.write(f"Noisy: mean = {noisy.mean():.2f}%, SD = {noisy.std():.2f}, n = {n2}\n\n")
    f.write(f"Shapiro-Wilk clean: W = {sw_clean.statistic:.3f}, p = {sw_clean.pvalue:.3f}\n")
    f.write(f"Shapiro-Wilk noisy: W = {sw_noisy.statistic:.3f}, p = {sw_noisy.pvalue:.3f}\n")
    f.write(f"Levene: W = {levene.statistic:.3f}, p = {levene.pvalue:.3f}\n\n")
    f.write(f"Welch's t-test: t = {t_test.statistic:.2f}, df = {t_test.df:.1f}, p = {t_test.pvalue:.6f}\n")
    f.write(f"Mean difference = {diff:.2f} percentage points\n")
    f.write(f"95% CI = [{ci.low:.2f}, {ci.high:.2f}]\n")
    f.write(f"Cohen's d = {d:.2f}\n\n")
    f.write(f"Mann-Whitney U: U = {mw.statistic:.1f}, p = {mw.pvalue:.6f}\n")
print("\nSaved results/stats_summary.txt")

# ----- Figure 1: example images, clean vs noisy -----
X, y = load_images()
X_noisy = add_noise(X, NOISE_SD, seed=0)

fig, axes = plt.subplots(2, 6, figsize=(8, 3))
for i in range(6):
    axes[0, i].imshow(X[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[1, i].imshow(X_noisy[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_title("Clean", loc="left")
axes[1, 0].set_title("Noisy (noise SD = 0.1)", loc="left")
plt.tight_layout()
plt.savefig("figures/example_images.png", dpi=150)
plt.close()

# ----- Figure 2: Q-Q plots to check normality visually -----
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
stats.probplot(clean, plot=axes[0])
axes[0].set_title("Q-Q plot: clean")
stats.probplot(noisy, plot=axes[1])
axes[1].set_title("Q-Q plot: noisy")
plt.tight_layout()
plt.savefig("figures/qq_plots.png", dpi=150)
plt.close()

# ----- Figure 3: accuracy for each run -----
plt.figure(figsize=(6, 4.5))
plt.boxplot([clean, noisy], showfliers=False)
plt.xticks([1, 2], ["Clean", "Noisy"])
# add the individual runs as dots (with a little sideways jitter so they don't overlap)
np.random.seed(0)
plt.scatter(np.random.normal(1, 0.04, n1), clean, color="#2a78d6", alpha=0.8)
plt.scatter(np.random.normal(2, 0.04, n2), noisy, color="#eb6834", alpha=0.8)
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy over 20 runs per condition")
plt.tight_layout()
plt.savefig("figures/accuracy_by_condition.png", dpi=150)
plt.close()

# ----- Figure 4: the difference with its 95% CI -----
plt.figure(figsize=(6, 2.5))
plt.errorbar(diff, 0, xerr=[[diff - ci.low], [ci.high - diff]], fmt="o", capsize=6, color="#2a78d6")
plt.axvline(0, color="black", linewidth=1)
plt.axvline(1, color="gray", linestyle="--", label="1-point drop (smallest effect we care about)")
plt.yticks([])
plt.xlabel("Drop in accuracy from noise (percentage points)")
plt.title("Mean difference with 95% CI")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("figures/difference_ci.png", dpi=150)
plt.close()
print("Saved figures to figures/")
