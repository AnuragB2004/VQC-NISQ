"""Datasets, stratified splits and leakage-free preprocessing shared by all experiments."""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEEDS = list(range(42, 57))          # 15 independent stratified train/test splits
CONFIGS = [(4, 2), (8, 2), (8, 4)]    # (n qubits, L layers)
DATASETS = ["breast_cancer", "digits"]
LABEL = {"breast_cancer": "Breast Cancer", "digits": "Binary digits"}
EPS_1Q = 1e-3
EPS_2Q_GRID = [1e-3, 1e-2, 1e-1]
EPS_2Q_PRIMARY = 1e-2
SHOTS = 1024
N_ITER = 20


def load(name):
    """Return X (raw features) and y in {-1, +1}."""
    if name == "breast_cancer":
        d = load_breast_cancer()          # 569 x 30; target 0 = malignant, 1 = benign
        X, t = d.data, d.target
    elif name == "digits":
        d = load_digits()                 # 8x8 images, 64 features
        mask = d.target <= 1              # digits 0 and 1 -> 178 + 182 = 360 samples
        X, t = d.data[mask], d.target[mask]
    else:
        raise ValueError(name)
    return X.astype(float), np.where(t == 1, 1, -1)


def split(X, y, seed):
    """Stratified 80/20 split; identical for the VQC and every classical model."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def preprocess(Xtr, Xte, dim):
    """Min-max -> PCA(dim) -> min-max to [0, 1]; every step fitted on the training partition only.
    Test features are clipped to [0, 1] so that R_Y(pi x) stays in the encoding range."""
    s1 = MinMaxScaler().fit(Xtr)
    pca = PCA(n_components=dim, random_state=0).fit(s1.transform(Xtr))
    Ztr, Zte = pca.transform(s1.transform(Xtr)), pca.transform(s1.transform(Xte))
    s2 = MinMaxScaler().fit(Ztr)
    return s2.transform(Ztr), np.clip(s2.transform(Zte), 0.0, 1.0)


def matched_data(name, seed, dim):
    X, y = load(name)
    Xtr, Xte, ytr, yte = split(X, y, seed)
    Ftr, Fte = preprocess(Xtr, Xte, dim)
    return Ftr, Fte, ytr, yte
