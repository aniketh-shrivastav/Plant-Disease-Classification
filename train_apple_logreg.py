import argparse
import json
from pathlib import Path
import numpy as np

# Reuse implementations from rhizome trainer (features, PCA, split, metrics, training)
import train_rhizome_logreg as rh


def parse_args():
    p = argparse.ArgumentParser(description="Train scratch Logistic Regression (healthy vs diseased) for Apple dataset using HOG+HSV (no sklearn)")
    p.add_argument("--data_dir", type=str, default=str(Path("apple")))
    p.add_argument("--output_dir", type=str, default=str(Path("outputs")/"apple_logreg"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--pca", action="store_true")
    p.add_argument("--pca_components", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--threshold_mode", type=str, default="balanced_acc", choices=["balanced_acc", "macro_f1", "healthy_recall"])
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Apple data (binary healthy vs diseased) from: {data_dir}")
    X, y = rh.build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    model, pca, scaler, metrics = rh.train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=args.pca,
        pca_components=args.pca_components,
        epochs=args.epochs,
        lr=args.lr,
        threshold_mode=args.threshold_mode,
    )

    # Save artifacts (same format as rhizome)
    np.save(out_dir / "weights.npy", model.weights)
    if pca is not None:
        np.savez(out_dir / "pca.npz", mean=pca["mean"], components=pca["components"])
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Brief summary
    print("Apple Logistic Regression training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train_samples", "n_test_samples", "n_features", "pca_components", "threshold"]}, indent=2))


if __name__ == "__main__":
    main()
