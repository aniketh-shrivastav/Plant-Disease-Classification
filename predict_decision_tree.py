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


def load_artifacts(model_path: Path, label_map_path: Path):
    model = joblib.load(model_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        id_to_str = json.load(f)
    # keys as strings possible; normalize to int
    id_to_str = {int(k): v for k, v in id_to_str.items()}
    return model, id_to_str


def main():
    ap = argparse.ArgumentParser(description="Predict a single image with the Decision Tree model")
    ap.add_argument("--image", required=True, type=str, help="Path to image to classify")
    ap.add_argument("--model", type=str, default=str(Path("outputs") / "tree" / "decision_tree.joblib"))
    ap.add_argument("--labels", type=str, default=str(Path("outputs") / "tree" / "label_map.json"))
    args = ap.parse_args()

    model_path = Path(args.model)
    label_map_path = Path(args.labels)
    img_path = Path(args.image)

    model, id_to_str = load_artifacts(model_path, label_map_path)
    x = extract_features(img_path).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        pred_id = int(np.argmax(probs))
        pred_label = id_to_str.get(pred_id, str(pred_id))
        print(f"Image: {img_path}")
        print(f"Prediction: {pred_label} (class_id={pred_id}, prob={probs[pred_id]:.4f})")
        # Show top-3
        top3 = np.argsort(probs)[-3:][::-1]
        print("Top-3:")
        for k in top3:
            print(f"  {id_to_str.get(int(k), str(int(k)))}: {probs[int(k)]:.4f}")
    else:
        pred_id = int(model.predict(x)[0])
        pred_label = id_to_str.get(pred_id, str(pred_id))
        print(f"Image: {img_path}")
        print(f"Prediction: {pred_label} (class_id={pred_id})")


if __name__ == "__main__":
    main()
