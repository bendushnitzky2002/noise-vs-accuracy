# Does a Little Noise Hurt a Model?

AIPI 510 – Module Project 2: Statistical Analysis (Ben Dushnitzky)

## Overview

Real images usually aren't perfectly clean. Phone cameras in low light, webcams and scanners all add some random "grain." For this project I tested whether a small amount of that kind of noise makes a simple model worse at recognizing handwritten digits.

**Research question:** Does adding light random noise to images change the test accuracy of a digit classifier?

- **H0:** the mean accuracy is the same for clean and noisy images (μ_clean − μ_noisy = 0)
- **H1:** the mean accuracy is different (μ_clean − μ_noisy ≠ 0)

**Result:** the noise lowered accuracy by about **1.11 percentage points** on average (95% CI 0.63 to 1.59, Welch's t-test p < 0.001, Cohen's d = 1.48).

## Experiment design

| | |
|---|---|
| Independent variable | Image condition: **clean** or **noisy** (Gaussian noise with SD = 0.1, pixels scaled 0–1) |
| Dependent variable | Test accuracy (% of test images the model got right) |
| Control condition | Clean images |
| Kept the same | Dataset, model (logistic regression), 70/30 train/test split, all settings |
| Experimental unit | One run = one random train/test split + one trained model |
| Randomization | 40 runs (20 per condition), done in a shuffled order, each with its own random seed |

How I picked the noise level: before the real experiment I tried a few levels (SD = 0.1, 0.2, 0.3 and 0.5) on some quick test runs. At 0.3 and up the accuracy dropped a lot (about 86% at 0.3 and 67% at 0.5), which seemed too obvious to test, so I went with 0.1. Those test runs aren't part of the data in this repo.

## Data

- **Dataset:** the handwritten digits dataset that comes with scikit-learn (`sklearn.datasets.load_digits`). It has 1,797 grayscale 8x8 images of the digits 0–9. It's a copy of the test set from the UCI "Optical Recognition of Handwritten Digits" dataset.
  - Citation: Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49
- **Raw data** (made by the scripts):
  - `data/raw/pilot_results.csv`: the pilot runs (10 per condition) used for the power analysis
  - `data/raw/experiment_results.csv`: the 40 main runs (run order, condition, seed, accuracy)
- **Cleaned data:** `data/clean/experiment_results_clean.csv`. I checked for missing values, duplicate seeds and impossible accuracies, and added accuracy as a percentage.

You don't need to download anything. The images come installed with scikit-learn.

## Project structure

```
src/
  experiment.py       # shared code: load images, add noise, train + test one model
  run_pilot.py        # 1. pilot runs to see how much accuracy varies
  power_analysis.py   # 2. power analysis -> how many runs per group we need
  run_experiment.py   # 3. main experiment (data collection) in random order
  clean_data.py       # 4. checks + cleaning
  analysis.py         # 5. assumption checks (Shapiro-Wilk, Q-Q plots, Levene), Welch's t-test, CI, effect size, figures
data/raw/             # raw results
data/clean/           # cleaned results
results/              # power_analysis.txt, stats_summary.txt
figures/              # charts for the report and presentation
```

## How to reproduce

Run these from the main folder of the repo (Python 3.10+):

```bash
pip install -r requirements.txt
python src/run_pilot.py        # -> data/raw/pilot_results.csv
python src/power_analysis.py   # -> results/power_analysis.txt, figures/power_curve.png
python src/run_experiment.py   # -> data/raw/experiment_results.csv
python src/clean_data.py       # -> data/clean/experiment_results_clean.csv
python src/analysis.py         # -> results/stats_summary.txt, figures/*.png
```

All the random seeds are fixed, so you should get the same numbers I did. The whole thing takes less than a minute.

## Power analysis

- I ran a pilot with 10 runs per condition. The pooled SD of accuracy was 0.76 percentage points.
- I decided the smallest drop I care about is 1 percentage point.
- That gives Cohen's d = 1.0 / 0.76 = 1.32. With α = 0.05, power = 0.80 and a two-sided test, I needed **11 runs per group**.
- I did 20 per group since each run only takes a second, so there's some extra margin in case the pilot underestimated the variation.

## Results

| | Clean | Noisy |
|---|---|---|
| Mean accuracy | 96.57% | 95.46% |
| SD | 0.59 | 0.88 |
| Runs | 20 | 20 |

- Shapiro-Wilk: no evidence against normality in either group (p = 0.25 and 0.13), and the Q-Q plots (`figures/qq_plots.png`) look pretty straight too
- Levene: no evidence the variances are different (p = 0.17). I used Welch's t-test anyway.
- Welch's t-test: t = 4.69, df = 33.4, p < 0.001
- Mean difference 1.11 points, 95% CI [0.63, 1.59], Cohen's d = 1.48
- Mann-Whitney U (backup that doesn't assume normality): p < 0.001

The full output is in `results/stats_summary.txt`.
