import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from skimage.feature import hog


def extract_color_histogram(img_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.concatenate(feats, axis=0)


def extract_hog(img_bgr: np.ndarray, resize=(128, 128)) -> np.ndarray:
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


def load_artifacts(model_dir: Path):
    d = np.load(model_dir / "svm_alpha.npz")
    alpha = d["alpha"]
    b = float(d["b"][0])
    X_sv = np.load(model_dir / "svm_X_sv.npy")
    y_sv = np.load(model_dir / "svm_y_sv.npy")
    pca = None
    if (model_dir / "pca.npz").exists():
        p = np.load(model_dir / "pca.npz")
        pca = {"mean": p["mean"], "components": p["components"]}
    scaler_mean = None
    scaler_std = None
    if (model_dir / "scaler.npz").exists():
        s = np.load(model_dir / "scaler.npz")
        scaler_mean = s["mean"]
        scaler_std = s["std"]
    thr = 0.0
    if (model_dir / "threshold.json").exists():
        with open(model_dir / "threshold.json", "r", encoding="utf-8") as f:
            thr = float(json.load(f).get("threshold", 0.0))
    kernel = "rbf"
    gamma = 0.05
    degree = 3
    if (model_dir / "metrics.json").exists():
        with open(model_dir / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
            kernel = m.get("kernel", kernel)
            gamma = m.get("gamma", gamma)
            degree = m.get("degree", degree)
    return alpha, b, X_sv, y_sv, pca, scaler_mean, scaler_std, thr, kernel, gamma, degree


def kernel_eval(x1: np.ndarray, x2: np.ndarray, kernel: str, gamma: float, degree: int) -> float:
    if kernel == "linear":
        return float(np.dot(x1, x2))
    elif kernel == "poly":
        return float((1.0 + np.dot(x1, x2)) ** degree)
    else:
        return float(np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2))


def main():
    ap = argparse.ArgumentParser(description="Predict Dragon Fruit disease with scratch Kernel SVM")
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--model_dir", type=str, default=str(Path("outputs")/"dragon_kernel_svm"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    alpha, b, X_sv, y_sv, pca, mean, std, thr, kernel, gamma, degree = load_artifacts(model_dir)

    x = extract_features(Path(args.image)).reshape(1, -1)
    if pca is not None:
        x = (x - pca["mean"].reshape(1, -1)) @ pca["components"].T
    if mean is not None and std is not None:
        x = (x - mean) / np.where(std == 0, 1.0, std)

    s = 0.0
    for a, y_i, x_sv in zip(alpha, y_sv, X_sv):
        if a > 0:
            s += a * y_i * kernel_eval(x_sv, x[0], kernel, gamma, degree)
    s += b
    margin = float(s)
    pred = 1 if margin >= thr else 0
    label = "diseased" if pred == 1 else "healthy"

    print(f"Image: {args.image}")
    print(f"Prediction: {label} (class_id={pred}, margin={margin:.4f}, threshold={thr:.4f}, kernel={kernel})")


if __name__ == "__main__":
    main()
