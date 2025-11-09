import argparse
import json
from pathlib import Path
import numpy as np

import train_rhizome_tree as rh
from decision_tree import DecisionTreeScratch


def parse_args():
    ap = argparse.ArgumentParser(description="Train scratch Decision Tree for Dragon Fruit (healthy vs diseased) using HOG+HSV")
    ap.add_argument("--data_dir", type=str, default=str(Path("Dragon Fruit (Pitahaya)")/"Original Images"))
    ap.add_argument("--output_dir", type=str, default=str(Path("outputs")/"dragon_tree"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=128)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_samples_split", type=int, default=5)
    ap.add_argument("--max_features", type=int, default=None)
    ap.add_argument("--max_thresholds", type=int, default=32)
    return ap.parse_args()


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Dragon Fruit data from: {data_dir}")
    X, y = rh.build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    train_idx, test_idx = rh.stratified_train_test_split(y, test_size=args.test_size, seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pca_mean = None
    pca_components_mat = None
    if args.pca and X_train.shape[1] > args.pca_components:
        pca_mean, pca_components_mat = rh.pca_fit(X_train, n_components=args.pca_components)
        X_train = rh.pca_transform(X_train, pca_mean, pca_components_mat)
        X_test = rh.pca_transform(X_test, pca_mean, pca_components_mat)

    max_features = args.max_features
    if max_features is None:
        max_features = int(np.sqrt(X_train.shape[1]))
        max_features = max(1, max_features)

    clf = DecisionTreeScratch(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        max_thresholds=args.max_thresholds,
        seed=args.seed,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float((y_test == y_pred).mean())
    prec_w, rec_w, f1_w = rh.precision_recall_f1_weighted(y_test, y_pred)

    with open(out_dir / "tree.json", "w", encoding="utf-8") as f:
        json.dump(to_py(clf.tree), f)
    if pca_components_mat is not None:
        np.savez(out_dir / "pca.npz", mean=pca_mean, components=pca_components_mat)
    metrics = {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "pca_components": int(pca_components_mat.shape[0]) if pca_components_mat is not None else None,
        "max_depth": int(args.max_depth),
        "min_samples_split": int(args.min_samples_split),
        "max_features": int(max_features),
        "max_thresholds": int(args.max_thresholds),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Dragon Fruit Decision Tree training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train", "n_test", "n_features", "pca_components", "max_depth"]}, indent=2))


if __name__ == "__main__":
    main()
