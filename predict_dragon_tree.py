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


def load_model(model_dir: Path):
    with open(model_dir / "tree.json", "r", encoding="utf-8") as f:
        tree = json.load(f)
    pca = None
    if (model_dir / "pca.npz").exists():
        d = np.load(model_dir / "pca.npz")
        pca = {"mean": d["mean"], "components": d["components"]}
    return tree, pca


def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((X - mean.reshape(1, -1)) @ components.T).astype(np.float32)


def predict_one_tree(x: np.ndarray, node):
    if not isinstance(node, dict):
        return int(node)
    f = node["feature"]
    t = node["threshold"]
    if float(x[f]) <= float(t):
        return predict_one_tree(x, node["left"])
    else:
        return predict_one_tree(x, node["right"])


def main():
    ap = argparse.ArgumentParser(description="Predict Dragon Fruit disease with scratch Decision Tree")
    ap.add_argument("--image", required=True, type=str)
    ap.add_argument("--model_dir", type=str, default=str(Path("outputs")/"dragon_tree"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tree, pca = load_model(model_dir)

    x = extract_features(Path(args.image)).reshape(1, -1)
    if pca is not None:
        x = pca_transform(x, pca["mean"], pca["components"])

    pred = predict_one_tree(x[0], tree)
    label = "diseased" if pred == 1 else "healthy"
    print(f"Image: {args.image}")
    print(f"Prediction: {label} (class_id={pred})")


if __name__ == "__main__":
    main()
