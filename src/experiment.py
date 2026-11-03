# Shared code for the experiment: load the images, add noise, train and test one model.
# The pilot and the main experiment both use run_once() so they work exactly the same way.

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Noise level for the "noisy" condition: random noise with a standard deviation of
# 10% of the pixel brightness range (pixels are scaled to 0-1)
NOISE_SD = 0.1
TEST_SIZE = 0.3


def load_images():
    # 1,797 grayscale 8x8 images of handwritten digits (0-9) that come with scikit-learn.
    # Pixel values go from 0 to 16, so divide by 16 to put them on a 0-1 scale.
    X, y = load_digits(return_X_y=True)
    return X / 16.0, y


def add_noise(X, noise_sd, seed):
    # Add Gaussian noise to every pixel, then clip back into the 0-1 range
    rng = np.random.default_rng(seed)
    noisy = X + rng.normal(0, noise_sd, size=X.shape)
    return np.clip(noisy, 0, 1)


def run_once(seed, condition):
    # One run = one random train/test split + one trained model.
    # condition is "clean" or "noisy". Returns test accuracy (0-1).
    X, y = load_images()
    if condition == "noisy":
        X = add_noise(X, NOISE_SD, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
