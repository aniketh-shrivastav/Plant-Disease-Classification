import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# Feature extraction (reuse from LR pipeline)
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
    hog_feat = extract_hog(img)
    col_feat = extract_color_histogram(img)
    return np.concatenate([hog_feat, col_feat], axis=0)


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
    parent = img_path.parent.name
    if parent.lower() == "plantvillage":
        parent = img_path.parent.parent.name
    return parent


def map_group(label: str, grouping: str) -> str:
    s = label.lower()
    if grouping == "binary":
        return "healthy" if "healthy" in s else "unhealthy"
    if grouping == "5class":
        if "healthy" in s:
            return "healthy"
        if "bacterial" in s:
            return "bacterial"
        # fungal bucket
        if (
            "early_blight" in s
            or "late_blight" in s
            or "leaf_mold" in s
            or "septoria_leaf_spot" in s
            or "target_spot" in s
        ):
            return "fungal"
        # viral bucket
        if ("mosaic_virus" in s) or ("yellowleaf__curl_virus" in s) or ("yellowleaf" in s and "virus" in s):
            return "viral"
        # pest/mite bucket
        if "spider_mites" in s or "two_spotted_spider_mite" in s or "spider_mite" in s:
            return "pest"
        # fallback
        return "other"
    if grouping == "4class":
        if "healthy" in s:
            return "healthy"
        if (
            "early_blight" in s
            or "late_blight" in s
            or "leaf_mold" in s
            or "septoria_leaf_spot" in s
            or "target_spot" in s
        ):
            return "fungal"
        if "bacterial" in s:
            return "bacterial"
        if ("mosaic_virus" in s) or ("yellowleaf" in s and "virus" in s) or ("spider_mite" in s) or ("spider_mites" in s) or ("two_spotted_spider_mite" in s):
            return "viral_pest"
        return "other"
    # original: return as-is
    return label


def build_dataset(
    data_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    grouping: str = "original",
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    # Optional subsample for speed
    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(paths):
        paths = list(rng.choice(paths, size=limit, replace=False))

    labels_str_raw: List[str] = [infer_label_from_path(p) for p in paths]
    # Apply grouping
    labels_str: List[str] = [map_group(s, grouping) for s in labels_str_raw]
    classes = sorted(sorted(set(labels_str)))
    str_to_id = {s: i for i, s in enumerate(classes)}
    id_to_str = {i: s for s, i in str_to_id.items()}
    y = np.array([str_to_id[s] for s in labels_str], dtype=np.int64)

    # Extract features
    X_list: List[np.ndarray] = []
    bad = 0
    for p in tqdm(paths, desc="Extracting features", unit="img"):
        try:
            X_list.append(extract_features(p))
        except Exception:
            bad += 1
            continue
    if bad:
        print(f"Warning: skipped {bad} images due to read/feature errors")
    if not X_list:
        raise RuntimeError("No features could be extracted.")

    X = np.vstack(X_list).astype(np.float32)

    # Align labels in case of failures
    if len(X) != len(y):
        y = y[: len(X)]

    return X, y, id_to_str


# -----------------------------
# Train/eval Decision Tree
# -----------------------------

def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    max_depth: int | None = 18,
    max_features: str | int | float | None = "sqrt",
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | Dict[int, float] | None = None,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = clf.predict(X_test)
    infer_time_total = time.time() - t1
    infer_time_per_sample = infer_time_total / max(1, len(y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "train_time_sec": float(train_time),
        "infer_time_per_sample_sec": float(infer_time_per_sample),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return clf, metrics


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a Decision Tree on PlantVillage images (multi-class)")
    p.add_argument("--data_dir", type=str, default=str(Path("archive") / "PlantVillage"))
    p.add_argument("--limit", type=int, default=None, help="Optional limit of images for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--output_dir", type=str, default=str(Path("outputs") / "tree"))
    p.add_argument("--grouping", type=str, default="original", choices=["original", "binary", "5class", "4class"], help="Label grouping taxonomy")
    p.add_argument("--max_depth", type=int, default=18)
    p.add_argument("--max_features", type=str, default="sqrt", choices=["sqrt", "log2", "None"]) 
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--balanced", action="store_true", help="Use class_weight='balanced'")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect/prepare data
    print(f"Loading data from: {data_dir}")
    X, y, id_to_str = build_dataset(data_dir, limit=args.limit, seed=args.seed, grouping=args.grouping)
    print(f"Loaded features: X={X.shape}, y={y.shape}, classes={len(id_to_str)}")

    # Map max_features option
    max_features = None if args.max_features == "None" else args.max_features
    class_weight = "balanced" if args.balanced else None

    # Train/eval
    model, metrics = train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        max_depth=args.max_depth,
        max_features=max_features,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight,
    )

    # Persist
    joblib.dump(model, out_dir / "decision_tree.joblib")
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(id_to_str, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "train_time_sec": metrics["train_time_sec"],
        "infer_time_per_sample_sec": metrics["infer_time_per_sample_sec"],
        "n_train": metrics["n_train_samples"],
        "n_test": metrics["n_test_samples"],
        "n_features": metrics["n_features"],
        "n_classes": metrics["n_classes"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)

    print("Decision Tree training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
