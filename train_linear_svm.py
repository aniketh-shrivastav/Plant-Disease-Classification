import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import hog
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from linear_svm import LinearSVM


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
        raise ValueError(f"Failed to read image: {img_path}")
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


def infer_label_from_path(img_path: Path, root: Path) -> str:
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

    labels_str: List[str] = [infer_label_from_path(p, data_dir) for p in paths]
    y = np.array([0 if "healthy" in s.lower() else 1 for s in labels_str], dtype=np.int64)

    X_list: List[np.ndarray] = []
    kept_y: List[int] = []
    bad = 0
    for i, p in enumerate(tqdm(paths, desc="Extracting features", unit="img")):
        try:
            X_list.append(extract_features(p))
            kept_y.append(int(y[i]))
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


def reduce_with_incremental_pca(X: np.ndarray, n_components: int = 128, batch_size: int = 512) -> Tuple[np.ndarray, IncrementalPCA]:
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    n = X.shape[0]
    for start in range(0, n, batch_size):
        ipca.partial_fit(X[start:start + batch_size])
    X_red = np.empty((n, n_components), dtype=np.float32)
    for start in range(0, n, batch_size):
        X_red[start:start + batch_size] = ipca.transform(X[start:start + batch_size]).astype(np.float32)
    return X_red, ipca


# -----------------------------
# Train/eval LinearSVM
# -----------------------------

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def find_best_threshold(y_true: np.ndarray, margins: np.ndarray, mode: str = "macro_f1") -> Tuple[float, Dict[str, float]]:
    # Scan thresholds to optimize selected criterion
    ts = np.linspace(float(margins.min()), float(margins.max()), num=101)
    best_t = 0.0
    best_score = -1.0
    best_stats: Dict[str, float] = {}
    for t in ts:
        y_pred = (margins > t).astype(int)
        if mode == "healthy_recall":
            # healthy label is 0
            # compute recall for class 0
            tp = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 0) & (y_pred == 1))
            score = tp / (tp + fn + 1e-12)
        elif mode == "healthy_precision":
            # precision for class 0 (pred 0)
            tp = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 1) & (y_pred == 0))
            score = tp / (tp + fp + 1e-12)
        elif mode == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    # compute metrics at best threshold
    y_best = (margins > best_t).astype(int)
    best_stats = {
        "macro_f1": float(f1_score(y_true, y_best, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_best)),
    }
    return best_t, best_stats


def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    use_pca: bool = True,
    pca_components: int = 128,
    epochs: int = 25,
    lr: float = 0.001,
    lambda_param: float = 0.01,
    threshold_mode: str = "macro_f1",
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pca_model = None
    if use_pca and pca_components and X_train.shape[1] > pca_components:
        # Reduce dimensionality for faster Python-loop training
        X_train, pca_model = reduce_with_incremental_pca(X_train, n_components=pca_components)
        X_test = pca_model.transform(X_test).astype(np.float32)

    # Standardize features (helps SVM training and balances HOG/HSV scales)
    X_train, mean, std = standardize_fit(X_train)
    X_test = standardize_apply(X_test, mean, std)

    model = LinearSVM(lr=lr, lambda_param=lambda_param, epochs=epochs)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    margins = X_test.dot(model.w) - model.b
    # threshold optimization to reduce healthy->unhealthy false positives
    best_threshold, threshold_stats = find_best_threshold(y_test, margins, mode=threshold_mode)
    y_pred = (margins > best_threshold).astype(int)
    # Map predictions from {-1,1} back to {0,1}
    # our y_pred already in {0,1}
    y_pred_bin = y_pred

    acc = accuracy_score(y_test, y_pred_bin)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_bin, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred_bin, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_bin)

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
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "pca_components": int(pca_components) if pca_model is not None else None,
        "threshold": float(best_threshold),
        "threshold_mode": threshold_mode,
        "threshold_stats": threshold_stats,
    }
    # return scaler params as well
    scaler = {"mean": mean, "std": std}
    return model, pca_model, scaler, metrics


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train scratch Linear SVM (binary healthy vs unhealthy) on HOG+HSV features")
    p.add_argument("--data_dir", type=str, default=str(Path("archive") / "PlantVillage"))
    p.add_argument("--output_dir", type=str, default=str(Path("outputs") / "linsvm"))
    p.add_argument("--limit", type=int, default=None, help="Optional limit of images for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--pca", action="store_true", help="Enable PCA for faster training")
    p.add_argument("--pca_components", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lambda_param", type=float, default=0.01)
    p.add_argument("--threshold_mode", type=str, default="macro_f1", choices=["macro_f1", "healthy_recall", "healthy_precision", "balanced_acc"], help="Criterion to select decision threshold")
   
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data (binary healthy vs unhealthy) from: {data_dir}")
    X, y = build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    model, pca_model, scaler, metrics = train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=args.pca,
        pca_components=args.pca_components,
        epochs=args.epochs,
        lr=args.lr,
        lambda_param=args.lambda_param,
        threshold_mode=args.threshold_mode,
    )

    # Persist artifacts
    # Save weights and bias
    np.save(out_dir / "w.npy", model.w)
    with open(out_dir / "b.json", "w", encoding="utf-8") as f:
        json.dump({"b": float(model.b)}, f)
    if pca_model is not None:
        joblib.dump(pca_model, out_dir / "ipca.joblib")
    # Save scaler
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    # Save threshold
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    # Save metrics
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
        }
    ]).to_csv(out_dir / "metrics_summary.csv", index=False)

    print("Linear SVM training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "train_time_sec", "n_train_samples", "n_test_samples", "n_features", "pca_components"]}, indent=2))


if __name__ == "__main__":
    main()
