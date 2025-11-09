import argparse
import json
from pathlib import Path
import numpy as np

import train_rhizome_svm as rh


def parse_args():
    ap = argparse.ArgumentParser(description="Train scratch Linear SVM for Apple (healthy vs diseased) using HOG+HSV")
    ap.add_argument("--data_dir", type=str, default=str(Path("apple")))
    ap.add_argument("--output_dir", type=str, default=str(Path("outputs")/"apple_svm"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--pca", action="store_true")
    ap.add_argument("--pca_components", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lambda_param", type=float, default=0.01)
    ap.add_argument("--threshold_mode", type=str, default="balanced_acc", choices=["balanced_acc", "macro_f1", "healthy_recall"])
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Apple data (binary healthy vs diseased) from: {data_dir}")
    X, y = rh.build_dataset_binary(data_dir, limit=args.limit, seed=args.seed)
    print(f"Features loaded: X={X.shape}, y={y.shape}")

    svm, pca, scaler, metrics = rh.train_and_eval(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed,
        use_pca=args.pca,
        pca_components=args.pca_components,
        epochs=args.epochs,
        lr=args.lr,
        lambda_param=args.lambda_param,
        threshold_mode=args.threshold_mode,
    )

    np.savez(out_dir / "svm_wb.npz", w=svm.w.astype(np.float32), b=np.array([svm.b], dtype=np.float32))
    np.savez(out_dir / "scaler.npz", mean=scaler["mean"], std=scaler["std"])
    if pca is not None:
        np.savez(out_dir / "pca.npz", mean=pca["mean"], components=pca["components"])
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": metrics["threshold"], "mode": metrics["threshold_mode"]}, f)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Apple Linear SVM training complete.")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1_weighted", "n_train", "n_test", "n_features", "pca_components", "threshold"]}, indent=2))


if __name__ == "__main__":
    main()
