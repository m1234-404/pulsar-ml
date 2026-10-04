"""Data loading, splitting and oversampling for the HTRU2 pulsar dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["ip_mean", "ip_std", "ip_kurtosis", "ip_skew",
            "dmsnr_mean", "dmsnr_std", "dmsnr_kurtosis", "dmsnr_skew"]


def load_htru2(path="data/HTRU_2.csv"):
    """UCI HTRU_2.csv has no header: 8 features then the class label (last column)."""
    df = pd.read_csv(path, header=None, names=FEATURES + ["label"])
    return df[FEATURES].values.astype(float), df["label"].values.astype(int)


def make_synthetic(n=6000, pulsar_frac=0.09, seed=0):
    """Fake data with the same shape as HTRU2. ONLY for smoke-testing the code."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pulsar_frac).astype(int)
    X = rng.normal(size=(n, 8)) * [25, 6, 1.5, 6, 30, 15, 4, 40] + [110, 47, 0.5, 1, 12, 26, 8, 100]
    X[y == 1] += [-60, -8, 3, 25, 50, 20, -4, 80]
    return X, y


def split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def oversample(X, y, seed=0):
    """Random minority oversampling -> ~50/50 classes. Apply to TRAINING data only."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
    idx = rng.permutation(np.concatenate([neg, pos, extra]))
    return X[idx], y[idx]
