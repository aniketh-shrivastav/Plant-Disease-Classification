import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import time

import numpy as np
import cv2
from skimage.feature import hog

from decision_tree import DecisionTreeScratch


# --------------- Feature extraction ---------------
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


# --------------- Data utils ---------------
def find_images(data_dir: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    files: List[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and "svn-" not in p.name.lower():
            files.append(p)
    return files


def infer_label_from_path(img_path: Path) -> str:
    parent = img_path.parent.name
    if parent.lower() == "plantvillage":
        parent = img_path.parent.parent.name
    return parent


def build_dataset_binary(data_dir: Path, limit: int | None = None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(paths):
        paths = list(rng.choice(paths, size=limit, replace=False))

    labels_str: List[str] = [infer_label_from_path(p) for p in paths]
    y_all = np.array([0 if "healthy" in s.lower() else 1 for s in labels_str], dtype=np.int64)

    X_list: List[np.ndarray] = []
    kept_y: List[int] = []
    bad = 0
    for i, p in enumerate(paths):
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


# --------------- PCA (NumPy) ---------------
def pca_fit(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(max(1, min(n_components, Vt.shape[0])))
    components = Vt[:k]
    return mean.astype(np.float32).ravel(), components.astype(np.float32)


def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((X - mean.reshape(1, -1)) @ components.T).astype(np.float32)


# --------------- Metrics (NumPy) ---------------
def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    cm = confusion_matrix_binary(y_true, y_pred)
    supports = cm.sum(axis=1).astype(float)
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


# --------------- Split ---------------
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


def parse_args():
    ap = argparse.ArgumentParser(description="Train a scratch Decision Tree for rhizome disease (healthy vs diseased) using HOG+HSV features")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=str(Path("outputs")/"rhizome_tree"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction before tree training")
    ap.add_argument("--pca_components", type=int, default=128)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_samples_split", type=int, default=5)
    ap.add_argument("--max_features", type=int, default=None, help="Number of features to consider at each split (default: None -> all; suggest sqrt)")
    ap.add_argument("--max_thresholds", type=int, default=32, help="Max thresholds per feature (quantile-based)")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rhizome data from: {data_dir}")
    X, y = build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    # Split
    train_idx, test_idx = stratified_train_test_split(y, test_size=args.test_size, seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Optional PCA
    pca_mean = None
    pca_components_mat = None
    if args.pca and X_train.shape[1] > args.pca_components:
        pca_mean, pca_components_mat = pca_fit(X_train, n_components=args.pca_components)
        X_train = pca_transform(X_train, pca_mean, pca_components_mat)
        X_test = pca_transform(X_test, pca_mean, pca_components_mat)

    # If max_features not set, use sqrt(n_features)
    max_features = args.max_features
    if max_features is None:
        max_features = int(np.sqrt(X_train.shape[1]))
        max_features = max(1, max_features)

    clf = DecisionTreeScratch(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        max_thresholds=args.max_thresholds,
        seed=args.seed,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    acc = float((y_test == y_pred).mean())
    prec_w, rec_w, f1_w = precision_recall_f1_weighted(y_test, y_pred)
    cm = [[int(v) for v in row] for row in (y_test.reshape(-1, 1) * 0).repeat(2, axis=1)]  # placeholder to keep shape
    # build actual cm
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_test, y_pred):
        cm[int(t)][int(p)] += 1

    # Save artifacts
    # Tree is a nested dict of primitives -> JSON serializable
    def to_py(obj):
        if isinstance(obj, dict):
            return {str(k): to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_py(v) for v in obj]
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

    with open(out_dir / "tree.json", "w", encoding="utf-8") as f:
        json.dump(to_py(clf.tree), f)
    if pca_components_mat is not None:
        np.savez(out_dir / "pca.npz", mean=pca_mean, components=pca_components_mat)
    metrics = {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "pca_components": int(pca_components_mat.shape[0]) if pca_components_mat is not None else None,
        "max_depth": int(args.max_depth),
        "min_samples_split": int(args.min_samples_split),
        "max_features": int(max_features),
        "max_thresholds": int(args.max_thresholds),
        "train_time_sec": float(train_time),
        "confusion_matrix": cm,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Rhizome Decision Tree training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train", "n_test", "n_features", "pca_components", "max_depth", "train_time_sec"]}, indent=2))


if __name__ == "__main__":
    main()
