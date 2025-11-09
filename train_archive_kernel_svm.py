import argparse
import json
from pathlib import Path
import numpy as np
import time

from kernel_svm import KernelSVM
from train_turmeric_kernel_svm import (
    build_dataset_binary,
    pca_fit,
    pca_transform,
    standardize_fit,
    standardize_apply,
    stratified_train_test_split,
    precision_recall_f1_weighted,
    confusion_matrix_binary,
    find_best_threshold,
)


def train_and_eval_archive(
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
    max_iters: int = 200,
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
    ap = argparse.ArgumentParser(description="Train Kernel SVM for Archive/PlantVillage (healthy vs diseased) using HOG+HSV")
    ap.add_argument("--data_dir", type=str, default=str(Path("archive")/"PlantVillage"))
    ap.add_argument("--output_dir", type=str, default=str(Path("outputs")/"archive_kernel_svm"))
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of images for quicker runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=128)
    ap.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--max_iters", type=int, default=200)
    ap.add_argument("--threshold_mode", type=str, default="balanced_acc", choices=["balanced_acc", "macro_f1", "healthy_recall"])
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Archive PlantVillage data (binary healthy vs diseased) from: {data_dir}")
    X, y = build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    svm, pca_art, scaler, metrics = train_and_eval_archive(
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

    np.savez(out_dir / "svm_alpha.npz", alpha=svm.alpha.astype(np.float32), b=np.array([svm.b], dtype=np.float32))
    np.save(out_dir / "svm_X_sv.npy", svm.X_sv.astype(np.float32))
    np.save(out_dir / "svm_y_sv.npy", svm.y_sv.astype(np.int8))
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    if (pca_art is not None):
        np.savez(out_dir / "pca.npz", mean=pca_art["mean"], components=pca_art["components"])
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Archive Kernel SVM training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train", "n_test", "n_features", "pca_components", "threshold"]}, indent=2))


if __name__ == "__main__":
    main()
