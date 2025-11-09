import argparse
import json
from pathlib import Path
import numpy as np
import cv2

from train_turmeric_kernel_svm import (
    extract_features_single,
    pca_transform,
    standardize_apply,
)


def load_artifacts(out_dir: Path):
    alpha_b = np.load(out_dir / "svm_alpha.npz")
    alpha = alpha_b["alpha"].astype(np.float32)
    b = float(alpha_b["b"][0])
    X_sv = np.load(out_dir / "svm_X_sv.npy").astype(np.float32)
    y_sv = np.load(out_dir / "svm_y_sv.npy").astype(np.int8)
    scaler = np.load(out_dir / "scaler.npz")
    mean, std = scaler["mean"].astype(np.float32), scaler["std"].astype(np.float32)
    pca = None
    pca_path = out_dir / "pca.npz"
    if pca_path.exists():
        p = np.load(pca_path)
        pca = (p["mean"].astype(np.float32), p["components"].astype(np.float32))
    thr = 0.0
    thr_path = out_dir / "threshold.json"
    if thr_path.exists():
        with open(thr_path, "r", encoding="utf-8") as f:
            thr = float(json.load(f)["threshold"])
    return alpha, b, X_sv, y_sv, (mean, std), pca, thr


def kernel_eval(x, X_sv, alpha, y_sv, kernel="rbf", gamma=0.05, degree=3):
    s = 0.0
    for a, y, x_sv in zip(alpha, y_sv, X_sv):
        if a > 0:
            if kernel == "linear":
                kval = float(np.dot(x_sv, x))
            elif kernel == "poly":
                kval = float((1.0 + np.dot(x_sv, x)) ** degree)
            else:
                kval = float(np.exp(-gamma * np.linalg.norm(x_sv - x) ** 2))
            s += a * (1 if y > 0 else -1) * kval
    return s


def parse_args():
    ap = argparse.ArgumentParser(description="Predict Apple image using trained Kernel SVM")
    ap.add_argument("image_path", type=str, help="Path to image to predict")
    ap.add_argument("--model_dir", type=str, default=str(Path("outputs")/"apple_kernel_svm"))
    ap.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf"])
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--degree", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    alpha, b, X_sv, y_sv, scaler, pca, thr = load_artifacts(model_dir)

    # extract features
    feat = extract_features_single(Path(args.image_path))
    if pca is not None:
        feat = pca_transform(feat[None, :], pca[0], pca[1])[0]
    feat = standardize_apply(feat[None, :], scaler[0], scaler[1])[0]

    s = kernel_eval(feat, X_sv, alpha, y_sv, kernel=args.kernel, gamma=args.gamma, degree=args.degree) + b
    pred = int(s >= thr)
    label = "healthy" if pred == 0 else "diseased"
    print(json.dumps({
        "score": float(s),
        "threshold": float(thr),
        "pred": int(pred),
        "label": label
    }, indent=2))


if __name__ == "__main__":
    main()
