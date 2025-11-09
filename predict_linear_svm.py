import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import joblib
import numpy as np
from skimage.feature import hog


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
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return np.concatenate([extract_hog(img), extract_color_histogram(img)], axis=0).astype(np.float32)


def load_model(model_dir: Path, w_path: Path | None, b_path: Path | None, pca_path: Path | None):
    if w_path is None:
        w_path = model_dir / "w.npy"
    if b_path is None:
        b_path = model_dir / "b.json"
    if pca_path is None and (model_dir / "ipca.joblib").exists():
        pca_path = model_dir / "ipca.joblib"

    w = np.load(w_path)
    with open(b_path, "r", encoding="utf-8") as f:
        b = float(json.load(f)["b"])

    pca = None
    if pca_path is not None and Path(pca_path).exists():
        pca = joblib.load(pca_path)
    # load scaler if exists
    scaler_mean = None
    scaler_std = None
    scaler_path = model_dir / "scaler.npz"
    if scaler_path.exists():
        data = np.load(scaler_path)
        scaler_mean = data["mean"]
        scaler_std = data["std"]
    # load threshold if exists
    threshold = 0.0
    thr_path = model_dir / "threshold.json"
    if thr_path.exists():
        with open(thr_path, "r", encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.0))
    return w, b, pca, scaler_mean, scaler_std, threshold


def main():
    ap = argparse.ArgumentParser(description="Predict healthy/unhealthy using scratch Linear SVM")
    ap.add_argument("--image", required=True, type=str, help="Path to image to classify")
    ap.add_argument("--model_dir", type=str, default=str(Path("outputs") / "linsvm"))
    ap.add_argument("--w", type=str, default=None, help="Optional path to w.npy")
    ap.add_argument("--b", type=str, default=None, help="Optional path to b.json")
    ap.add_argument("--pca", type=str, default=None, help="Optional path to ipca.joblib (auto-detected if present)")
    args = ap.parse_args()

    img_path = Path(args.image)
    model_dir = Path(args.model_dir)
    w, b, pca, mean, std, threshold = load_model(model_dir, Path(args.w) if args.w else None, Path(args.b) if args.b else None, Path(args.pca) if args.pca else None)

    x = extract_features(img_path).reshape(1, -1)
    if pca is not None:
        x = pca.transform(x).astype(np.float32)
    if mean is not None and std is not None:
        std_adj = np.where(std == 0, 1.0, std)
        x = ((x - mean) / std_adj).astype(np.float32)

    # decision: margin = x·w - b (consistent with training code)
    margin = float(x.dot(w).ravel()[0] - b)
    pred = 1 if margin > threshold else 0  # 1=unhealthy, 0=healthy
    label = "unhealthy" if pred == 1 else "healthy"
    # Optional sigmoid for a readable score (not a calibrated probability)
    prob_unhealthy = 1.0 / (1.0 + np.exp(-margin))

    print(f"Image: {img_path}")
    print(f"Prediction: {label} (class_id={pred}, score_unhealthy={prob_unhealthy:.4f}, margin={margin:.4f}, threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
