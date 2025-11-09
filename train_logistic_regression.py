import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import hog
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Feature extraction
# -----------------------------

def extract_color_histogram(img_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """HSV color histogram normalized per channel -> 3*bins features."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.concatenate(feats, axis=0)


def extract_hog(img_bgr: np.ndarray, resize: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """HOG features from grayscale resized image."""
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
    try:
        hog_feat = extract_hog(img)
        col_feat = extract_color_histogram(img)
        return np.concatenate([hog_feat, col_feat], axis=0)
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed for {img_path}: {e}")


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


def build_dataset(
    data_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    rng = np.random.default_rng(seed)
    if limit is not None and limit < len(paths):
        paths = list(rng.choice(paths, size=limit, replace=False))

    labels_str: List[str] = []
    for p in paths:
        labels_str.append(infer_label_from_path(p, data_dir))

    if binary:
        y = np.array([0 if "healthy" in s.lower() else 1 for s in labels_str], dtype=np.int64)
        id_to_str = {0: "healthy", 1: "unhealthy"}
    else:
        classes = sorted(sorted(set(labels_str)))
        str_to_id = {s: i for i, s in enumerate(classes)}
        id_to_str = {i: s for s, i in str_to_id.items()}
        y = np.array([str_to_id[s] for s in labels_str], dtype=np.int64)

    X_list: List[np.ndarray] = []
    bad = 0
    for p in tqdm(paths, desc="Extracting features", unit="img"):
        try:
            X_list.append(extract_features(p))
        except Exception:
            bad += 1
            continue
    if bad:
        print(f"Warning: skipped {bad} images due to read/feature errors", file=sys.stderr)
    if not X_list:
        raise RuntimeError("No features could be extracted.")

    X = np.vstack(X_list).astype(np.float32)
    if len(X) != len(y):
        y = y[: len(X)]

    return X, y, id_to_str


def build_dataset_incremental(
    data_dir: Path,
    seed: int = 42,
    binary: bool = False,
    pca_components: int | None = None,
    cv_folds: int = 3,
    test_size: float = 0.2,
    batch_size: int = 512,
    work_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], object | None]:
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(paths))
    rng.shuffle(idxs)
    paths = [paths[i] for i in idxs]

    labels_str: List[str] = [infer_label_from_path(p, data_dir) for p in paths]

    if binary:
        y_all = np.array([0 if "healthy" in s.lower() else 1 for s in labels_str], dtype=np.int64)
        id_to_str = {0: "healthy", 1: "unhealthy"}
    else:
        classes = sorted(sorted(set(labels_str)))
        str_to_id = {s: i for i, s in enumerate(classes)}
        id_to_str = {i: s for s, i in str_to_id.items()}
        y_all = np.array([str_to_id[s] for s in labels_str], dtype=np.int64)

    feat_dim = None
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            f = np.concatenate([extract_hog(img), extract_color_histogram(img)], axis=0)
            feat_dim = int(f.shape[0])
            break
        except Exception:
            continue
    if feat_dim is None:
        raise RuntimeError("No features could be extracted from any image.")

    if work_dir is None:
        work_dir = Path("outputs") / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(paths)
    raw_mm_path = work_dir / f"X_raw_{int(time.time())}.dat"
    X_raw = np.memmap(raw_mm_path, dtype=np.float32, mode="w+", shape=(n_total, feat_dim))
    y_raw = np.empty((n_total,), dtype=np.int64)

    write_ptr = 0
    bad = 0
    for i, p in enumerate(tqdm(paths, desc="Extracting features (stream)", unit="img")):
        try:
            feats = extract_features(p).astype(np.float32)
            X_raw[write_ptr] = feats
            y_raw[write_ptr] = y_all[i]
            write_ptr += 1
        except Exception:
            bad += 1
            continue

    n = write_ptr
    if bad:
        print(f"Warning: skipped {bad} images due to read/feature errors", file=sys.stderr)
    if n == 0:
        raise RuntimeError("No features could be extracted.")

    X_view = X_raw[:n]
    y = y_raw[:n]
    pca_model = None

    if pca_components is not None and pca_components > 0:
        train_size = int(np.floor(n * (1.0 - test_size)))
        fold_train_size = int(np.floor(train_size * (cv_folds - 1) / cv_folds)) if cv_folds > 1 else train_size
        max_comp = max(1, min(fold_train_size - 1, feat_dim))
        eff_comp = max(1, min(int(pca_components), int(max_comp)))

        ipca = IncrementalPCA(n_components=eff_comp, batch_size=batch_size)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            ipca.partial_fit(X_view[start:end])

        red_mm_path = work_dir / f"X_reduced_{eff_comp}_{int(time.time())}.dat"
        X_red = np.memmap(red_mm_path, dtype=np.float32, mode="w+", shape=(n, eff_comp))
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            X_red[start:end] = ipca.transform(X_view[start:end]).astype(np.float32)

        del X_raw
        X_view = X_red
        pca_model = ipca

    return X_view, y, id_to_str, pca_model


# -----------------------------
# Training/evaluation (clean version)
# -----------------------------

def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    class_weight: str | None = "balanced",
    use_pca: bool = False,
    pca_components: int | None = None,
    grid_search: bool = False,
    cv_folds: int = 5,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    n_classes = len(np.unique(y))
    if n_classes == 2:
        solver = "liblinear"
        n_jobs = 1
    else:
        solver = "saga"
        n_jobs = os.cpu_count() or 1

    steps: List[tuple] = [("scaler", StandardScaler())]

    if use_pca and pca_components:
        fold_train_size = int(np.floor(X_train.shape[0] * (cv_folds - 1) / cv_folds)) if grid_search else X_train.shape[0]
        max_comp = max(1, min(fold_train_size - 1, X_train.shape[1]))
        eff_comp = max(1, min(int(pca_components), int(max_comp)))

        steps.append((
            "pca",
            PCA(n_components=eff_comp, svd_solver="randomized", whiten=True, random_state=seed),
        ))

    steps.append((
        "clf",
        LogisticRegression(
            max_iter=3000,
            solver=solver,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=seed,
        ),
    ))

    pipe = Pipeline(steps=steps)

    t0 = time.time()
    if grid_search:
        if solver == "liblinear":
            param_grid = {
                "clf__C": [0.01, 0.1, 0.3, 1, 3, 10],
                "clf__penalty": ["l2", "l1"],
            }
        else:
            param_grid = {"clf__C": [0.1, 1, 3, 10], "clf__penalty": ["l2"]}

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scoring = "f1" if n_classes == 2 else "f1_weighted"

        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        train_time = time.time() - t0
        pipe = search.best_estimator_
        best_params = search.best_params_
    else:
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        best_params = None

    t1 = time.time()
    y_pred = pipe.predict(X_test)
    infer_time_total = time.time() - t1
    infer_time_per_sample = infer_time_total / len(y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "train_time_sec": float(train_time),
        "infer_time_per_sample_sec": float(infer_time_per_sample),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if best_params is not None:
        result["best_params"] = best_params

    return pipe, result


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Logistic Regression on PlantVillage images using HOG+Color features")
    p.add_argument("--data_dir", type=str, default=str(Path("archive") / "PlantVillage"), help="Path to dataset root")
    p.add_argument("--limit", type=int, default=None, help="Optional limit of images for quick runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--binary", action="store_true", help="Train binary model: healthy vs unhealthy")
    p.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction before LR")
    p.add_argument("--pca_components", type=int, default=256, help="Number of PCA components when --pca is enabled")
    p.add_argument("--grid", action="store_true", help="Enable GridSearchCV over C and penalty")
    p.add_argument("--cv_folds", type=int, default=5, help="CV folds for grid search")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for streaming/IncrementalPCA")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    downstream_use_pca = args.pca
    saved_pca_path = None

    try:
        X, y, id_to_str = build_dataset(data_dir, limit=args.limit, seed=args.seed, binary=args.binary)
        print(f"Loaded features: X={X.shape}, y={y.shape}, classes={len(id_to_str)}")
    except (np.core._exceptions._ArrayMemoryError, MemoryError):
        print("MemoryError during in-memory feature assembly. Falling back to streaming with IncrementalPCA...", file=sys.stderr)
        X, y, id_to_str, pca_model = build_dataset_incremental(
            data_dir,
            seed=args.seed,
            binary=args.binary,
            pca_components=(args.pca_components if args.pca else None),
            cv_folds=args.cv_folds,
            test_size=args.test_size,
            batch_size=args.batch_size,
            work_dir=Path(args.output_dir) / "tmp",
        )
        downstream_use_pca = False
        if pca_model is not None:
            saved_pca_path = Path(args.output_dir) / "ipca.joblib"
            joblib.dump(pca_model, saved_pca_path)
        print(f"Loaded features (streamed): X={X.shape}, y={y.shape}, classes={len(id_to_str)}")

    model, metrics = train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=downstream_use_pca,
        pca_components=(args.pca_components if downstream_use_pca else None),
        grid_search=args.grid,
        cv_folds=args.cv_folds,
    )

    joblib.dump(model, out_dir / "logreg_model.joblib")
    if saved_pca_path is not None and saved_pca_path.exists():
        print(f"Saved IncrementalPCA to {saved_pca_path}")
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

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
