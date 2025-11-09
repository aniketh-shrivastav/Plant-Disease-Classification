import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from kernel_svm import KernelSVM
from train_turmeric_kernel_svm import (
    extract_features,
    pca_fit,
    pca_transform,
    standardize_fit,
    standardize_apply,
    stratified_train_test_split,
    precision_recall_f1_weighted,
    confusion_matrix_binary,
    find_best_threshold,
)


# -----------------------------
# Data loading (Potato-only)
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


def build_potato_dataset_binary(
    data_dir: Path,
    limit: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a binary dataset (healthy=0, diseased=1) restricted to Potato classes only."""
    all_paths = find_images(data_dir)
    if not all_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    # Filter to Potato classes only (e.g., Potato___healthy, Potato___Early_blight, Potato___Late_blight)
    paths: List[Path] = []
    for p in all_paths:
        lab = infer_label_from_path(p)
        if lab.startswith("Potato___"):
            paths.append(p)

    if not paths:
        raise FileNotFoundError(
            "No Potato images found. Expected folders like 'Potato___healthy', 'Potato___Early_blight', 'Potato___Late_blight'."
        )

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


# -----------------------------
# Train/Eval
# -----------------------------

def train_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
    use_pca: bool = True,
    pca_components: int = 128,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float = 0.05,
    degree: int = 3,
    max_iters: int = 300,
    threshold_mode: str = "balanced_acc",
):
    train_idx, test_idx = stratified_train_test_split(y, test_size=test_size, seed=seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pca_mean = None
    pca_comp = None
    if use_pca and pca_components and X_train.shape[1] > pca_components:
        pca_mean, pca_comp = pca_fit(X_train, n_components=pca_components)
        X_train = pca_transform(X_train, pca_mean, pca_comp)
        X_test = pca_transform(X_test, pca_mean, pca_comp)

    X_train, mean, std = standardize_fit(X_train)
    X_test = standardize_apply(X_test, mean, std)

    svm = KernelSVM(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iters=max_iters)
    t0 = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - t0

    # Compute margins on test set using trained support vectors
    y_train_svm = np.where(y_train <= 0, -1, 1)
    margins = []
    for x in X_test:
        s = 0.0
        for a, y_sv, x_sv in zip(svm.alpha, y_train_svm, X_train):
            if a > 0:
                if kernel == "linear":
                    kval = float(np.dot(x_sv, x))
                elif kernel == "poly":
                    kval = float((1.0 + np.dot(x_sv, x)) ** degree)
                else:
                    kval = float(np.exp(-gamma * np.linalg.norm(x_sv - x) ** 2))
                s += a * y_sv * kval
        s += svm.b
        margins.append(s)
    margins = np.array(margins, dtype=np.float32)

    thr, thr_stats = find_best_threshold(y_test, margins, mode=threshold_mode)
    y_pred = (margins >= thr).astype(int)

    acc = float((y_test == y_pred).mean())
    prec_w, rec_w, f1_w = precision_recall_f1_weighted(y_test, y_pred)
    cm = confusion_matrix_binary(y_test, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "train_time_sec": float(train_time),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "pca_components": int(pca_comp.shape[0]) if pca_comp is not None else None,
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "degree": degree,
        "max_iters": int(max_iters),
        "threshold": float(thr),
        "threshold_mode": threshold_mode,
        "threshold_stats": thr_stats,
        "confusion_matrix": cm,
    }

    scaler = {"mean": mean, "std": std}
    pca_art = None
    if pca_comp is not None:
        pca_art = {"mean": pca_mean, "components": pca_comp}
    return svm, pca_art, scaler, metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Train Kernel SVM (Potato-only healthy vs diseased) using HOG+HSV features")
    ap.add_argument("--data_dir", type=str, default=str(Path("archive")/"PlantVillage"))
    ap.add_argument("--output_dir", type=str, default=str(Path("outputs")/"potato_kernel_svm"))
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of images for quicker runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=128)
    ap.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max_iters", type=int, default=300)
    ap.add_argument("--threshold_mode", type=str, default="balanced_acc", choices=["balanced_acc", "macro_f1", "healthy_recall"])  # training-time selection only
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Potato-only data (binary healthy vs diseased) from: {data_dir}")
    X, y = build_potato_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    svm, pca_art, scaler, metrics = train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=args.pca,
        pca_components=args.pca_components,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        max_iters=args.max_iters,
        threshold_mode=args.threshold_mode,
    )

    # Save artifacts (support vectors + scaler + optional PCA + threshold + metrics)
    np.savez(out_dir / "svm_alpha.npz", alpha=svm.alpha.astype(np.float32), b=np.array([svm.b], dtype=np.float32))
    np.save(out_dir / "svm_X_sv.npy", svm.X_sv.astype(np.float32))
    np.save(out_dir / "svm_y_sv.npy", svm.y_sv.astype(np.int8))
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    if pca_art is not None:
        np.savez(out_dir / "pca.npz", mean=pca_art["mean"], components=pca_art["components"])
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Potato Kernel SVM training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train", "n_test", "n_features", "pca_components", "kernel", "C", "gamma", "threshold"]}, indent=2))


if __name__ == "__main__":
    main()
