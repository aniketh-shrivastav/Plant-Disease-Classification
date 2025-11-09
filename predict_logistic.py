import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Reuse the same feature logic as in training

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


def extract_features(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return np.concatenate([extract_hog(img), extract_color_histogram(img)], axis=0).astype(np.float32)


def load_artifacts(model_path: Path, label_map_path: Path, pca_path: Path | None = None):
    model = joblib.load(model_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        id_to_str = {int(k): v for k, v in json.load(f).items()}
    str_to_id = {v: k for k, v in id_to_str.items()}
    pca_model = None
    if pca_path is not None and pca_path.exists():
        pca_model = joblib.load(pca_path)
    else:
        # Also try default location next to model if pca_path not specified
        default_ipca = model_path.parent / "ipca.joblib"
        if default_ipca.exists():
            pca_model = joblib.load(default_ipca)
    return model, id_to_str, str_to_id, pca_model


def predict_one(model, id_to_str, img_path: Path, pca_model=None):
    x = extract_features(img_path)[None, :]
    # If a PCA model was persisted during training (streaming mode), apply it
    if pca_model is not None:
        x = pca_model.transform(x)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        pred_id = int(np.argmax(probs))
        pred_prob = float(probs[pred_id])
    else:
        pred_id = int(model.predict(x)[0])
        pred_prob = None
    return pred_id, id_to_str[pred_id], pred_prob


def find_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def infer_label_from_path(img_path: Path) -> str:
    parent = img_path.parent.name
    if parent.lower() == "plantvillage":
        parent = img_path.parent.parent.name
    return parent


def main():
    ap = argparse.ArgumentParser(description="Predict using saved Logistic Regression model")
    ap.add_argument("--model", type=str, default="outputs/logreg_model.joblib")
    ap.add_argument("--labels", type=str, default="outputs/label_map.json")
    ap.add_argument("--image", type=str, help="Path to a single image to predict")
    ap.add_argument("--dir", type=str, help="If provided, run predictions on all images in this folder recursively")
    ap.add_argument("--metrics", action="store_true", help="When using --dir, compute metrics using folder names as ground truth")
    ap.add_argument("--pca", type=str, default=None, help="Optional path to saved PCA model (ipca.joblib)")
    args = ap.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    pca_path = Path(args.pca) if args.pca else None
    model, id_to_str, str_to_id, pca_model = load_artifacts(model_path, labels_path, pca_path)

    if args.image:
        img_path = Path(args.image)
        pred_id, pred_label, pred_prob = predict_one(model, id_to_str, img_path, pca_model)
        print(f"Image: {img_path}")
        if pred_prob is None:
            print(f"Prediction: {pred_label} (class_id={pred_id})")
        else:
            print(f"Prediction: {pred_label} (class_id={pred_id}, prob={pred_prob:.4f})")
        return

    if args.dir:
        root = Path(args.dir)
        paths = find_images(root)
        y_true = []
        y_pred = []
        for p in tqdm(paths, desc="Predicting", unit="img"):
            pred_id, pred_label, pred_prob = predict_one(model, id_to_str, p, pca_model)
            print(f"{p}\t{pred_label}\t{pred_prob if pred_prob is not None else ''}")
            if args.metrics:
                true_label = infer_label_from_path(p)
                if true_label in str_to_id:
                    y_true.append(str_to_id[true_label])
                    y_pred.append(pred_id)
        if args.metrics and y_true:
            acc = accuracy_score(y_true, y_pred)
            print(f"\nAccuracy: {acc:.4f}")
            print("Classification report:")
            cr = classification_report(y_true, y_pred, zero_division=0)
            print(cr)
            print("Confusion matrix:")
            print(confusion_matrix(y_true, y_pred))
        return

    print("Specify --image <path> or --dir <folder>.")


if __name__ == "__main__":
    main()
