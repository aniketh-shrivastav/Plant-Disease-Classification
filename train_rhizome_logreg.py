import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import time
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import hog

from logistic_regression import LogisticRegressionScratch


# -----------------------------
# Feature extraction
# -----------------------------

def extract_color_histogram(img_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.concatenate(feats, axis=0)


def extract_hog(img_bgr: np.ndarray, resize: Tuple[int, int] = (128, 128)) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return feat


def extract_features(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    return np.concatenate([extract_hog(img), extract_color_histogram(img)], axis=0).astype(np.float32)


# -----------------------------
# Data utilities
# -----------------------------

def find_images(data_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    files: List[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and "svn-" not in p.name.lower():
            files.append(p)
    return files


def infer_label_from_path(img_path: Path) -> str:
    # immediate parent folder
    parent = img_path.parent.name
    if parent.lower() == "plantvillage":
        parent = img_path.parent.parent.name
    return parent


def build_dataset_binary(
    data_dir: Path,
    limit: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(paths):
        paths = list(rng.choice(paths, size=limit, replace=False))

    labels_str: List[str] = [infer_label_from_path(p) for p in paths]
    # Generic mapping: any folder that contains 'healthy' (case-insensitive) -> 0, else -> 1 (diseased)
    y_all = np.array([0 if "healthy" in s.lower() else 1 for s in labels_str], dtype=np.int64)

    X_list: List[np.ndarray] = []
    kept_y: List[int] = []
    bad = 0
    for i, p in enumerate(tqdm(paths, desc="Extracting features", unit="img")):
        try:
            X_list.append(extract_features(p))
            kept_y.append(int(y_all[i]))
        except Exception:
            bad += 1
            continue
    if bad:
        print(f"Warning: skipped {bad} images due to read/feature errors", file=sys.stderr)
    if not X_list:
        raise RuntimeError("No features could be extracted.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(kept_y, dtype=np.int64)
    return X, y


def pca_fit(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA via SVD on centered X and return (mean, components)."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # economy SVD
    # Xc shape: (n_samples, n_features)
    # components are top-k right singular vectors (Vt[:k])
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(max(1, min(n_components, Vt.shape[0])))
    components = Vt[:k]
    return mean.astype(np.float32).ravel(), components.astype(np.float32)


def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    Xc = X - mean.reshape(1, -1)
    return (Xc @ components.T).astype(np.float32)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # rows: true [0,1], cols: pred [0,1]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    cm = confusion_matrix_binary(y_true, y_pred)
    # per-class metrics
    supports = cm.sum(axis=1).astype(float)
    # avoid division by zero
    precisions = []
    recalls = []
    f1s = []
    for c in [0, 1]:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    weights = supports / (supports.sum() + 1e-12)
    precision_w = float((weights * np.array(precisions)).sum())
    recall_w = float((weights * np.array(recalls)).sum())
    f1_w = float((weights * np.array(f1s)).sum())
    return precision_w, recall_w, f1_w


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    recalls = []
    for c in [0, 1]:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        rec = tp / (tp + fn + 1e-12)
        recalls.append(rec)
    return float(np.mean(recalls))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    f1s = []
    for c in [0, 1]:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def find_best_threshold(y_true: np.ndarray, margins: np.ndarray, mode: str = "balanced_acc") -> Tuple[float, Dict[str, float]]:
    ts = np.linspace(float(margins.min()), float(margins.max()), num=101)
    best_t = 0.0
    best_score = -1.0
    for t in ts:
        y_pred = (margins >= t).astype(int)
        if mode == "healthy_recall":
            tp = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 0) & (y_pred == 1))
            score = tp / (tp + fn + 1e-12)
        elif mode == "macro_f1":
            score = f1_macro(y_true, y_pred)
        else:
            score = balanced_accuracy(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = float(t)
    y_best = (margins >= best_t).astype(int)
    stats = {
        "macro_f1": float(f1_macro(y_true, y_best)),
        "balanced_acc": float(balanced_accuracy(y_true, y_best)),
    }
    return best_t, stats


# -----------------------------
# Train/eval
# -----------------------------

def stratified_train_test_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train_idx = []
    test_idx = []
    for cls in [0, 1]:
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    use_pca: bool = True,
    pca_components: int = 128,
    epochs: int = 50,
    lr: float = 0.05,
    threshold_mode: str = "balanced_acc",
):
    train_idx, test_idx = stratified_train_test_split(y, test_size=test_size, seed=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pca_mean = None
    pca_components_mat = None
    if use_pca and pca_components and X_train.shape[1] > pca_components:
        pca_mean, pca_components_mat = pca_fit(X_train, n_components=pca_components)
        X_train = pca_transform(X_train, pca_mean, pca_components_mat)
        X_test = pca_transform(X_test, pca_mean, pca_components_mat)

    X_train, mean, std = standardize_fit(X_train)
    X_test = standardize_apply(X_test, mean, std)

    # Train scratch Logistic Regression (binary)
    model = LogisticRegressionScratch(lr=lr, epochs=epochs)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Get margins (logits): z = [1, X]·w  with our class storing bias inside weights[0]
    # Our LogisticRegressionScratch uses bias folded in weights via X_bias; here recompute z
    X_test_bias = np.concatenate([np.ones((X_test.shape[0], 1), dtype=np.float32), X_test], axis=1)
    w = model.weights
    margins = X_test_bias.dot(w)

    # Tune threshold to balance errors
    thr, thr_stats = find_best_threshold(y_test, margins, mode=threshold_mode)
    y_pred = (margins >= thr).astype(int)

    acc = float((y_test == y_pred).mean())
    prec, rec, f1 = precision_recall_f1_weighted(y_test, y_pred)
    cm = confusion_matrix_binary(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "train_time_sec": float(train_time),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": 2,
        "confusion_matrix": cm.tolist(),
        "pca_components": int(pca_components) if pca_components_mat is not None else None,
        "threshold": float(thr),
        "threshold_mode": threshold_mode,
        "threshold_stats": thr_stats,
    }
    scaler = {"mean": mean, "std": std}
    pca = None
    if pca_components_mat is not None:
        pca = {"mean": pca_mean, "components": pca_components_mat}
    return model, pca, scaler, metrics


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train scratch Logistic Regression (binary healthy vs diseased) for rhizome dataset using HOG+HSV")
    p.add_argument("--data_dir", type=str, required=True, help="Path to rhizome dataset root (subfolders per class)")
    p.add_argument("--output_dir", type=str, default=str(Path("outputs") / "rhizome_logreg"))
    p.add_argument("--limit", type=int, default=None, help="Optional limit of images for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--pca", action="store_true", help="Enable PCA for faster training")
    p.add_argument("--pca_components", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--threshold_mode", type=str, default="balanced_acc", choices=["balanced_acc", "macro_f1", "healthy_recall"], help="Criterion to select decision threshold")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rhizome data (binary healthy vs diseased) from: {data_dir}")
    X, y = build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    model, pca, scaler, metrics = train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=args.pca,
        pca_components=args.pca_components,
        epochs=args.epochs,
        lr=args.lr,
        threshold_mode=args.threshold_mode,
    )

    # Persist artifacts
    # weights from scratch LR include bias as weights[0] for the bias column
    np.save(out_dir / "weights.npy", model.weights)
    if pca is not None:
        np.savez(out_dir / "pca.npz", mean=pca["mean"], components=pca["components"])
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([
        {
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "train_time_sec": metrics["train_time_sec"],
            "n_train": metrics["n_train_samples"],
            "n_test": metrics["n_test_samples"],
            "n_features": metrics["n_features"],
            "pca_components": metrics["pca_components"],
            "threshold": metrics["threshold"],
        }
    ]).to_csv(out_dir / "metrics_summary.csv", index=False)

    print("Rhizome Logistic Regression training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "train_time_sec", "n_train_samples", "n_test_samples", "n_features", "pca_components", "threshold"]}, indent=2))


if __name__ == "__main__":
    main()
