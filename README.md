# PlantVillage - Classical ML Baseline (Logistic Regression)

This repo trains a Logistic Regression classifier on PlantVillage leaf images using classic features (HOG + HSV color histograms). It reports accuracy, precision, recall, F1, and simple compute metrics (train time and per-sample inference time).

## Dataset

Expected root: `archive/PlantVillage/` (your workspace already has this). The script finds images recursively and infers labels from the parent folder name. It skips folders named `PlantVillage` as ambiguous intermediates and any folders starting with `svn-`.

## Quick start

Run a quick smoke test on a subset (fast):

```
C:/Python313/python.exe train_logistic_regression.py --limit 300 --test_size 0.2 --seed 42
```

Full run (can take a while depending on CPU):

```
C:/Python313/python.exe train_logistic_regression.py --test_size 0.2 --seed 42
```

### Binary mode (healthy vs unhealthy)

To collapse all classes into two labels (healthy/unhealthy) and usually get higher accuracy:

```
C:/Python313/python.exe train_logistic_regression.py --binary --limit 300 --test_size 0.2 --seed 42
```

In binary mode, the label map becomes `{0: "healthy", 1: "unhealthy"}` and the model uses a faster solver.

### Improve accuracy (PCA + Grid Search)

Enable PCA to reduce 8K+ features to a compact space and try a small hyperparameter sweep:

```
C:/Python313/python.exe train_logistic_regression.py --binary --pca --pca_components 256 --grid --cv_folds 5 --test_size 0.2 --seed 42 --limit 600
```

For the best results, remove `--limit` and train on the full dataset.

Artifacts will be written to `outputs/`:

- `logreg_model.joblib` – trained sklearn Pipeline (StandardScaler + LogisticRegression)
- `label_map.json` – id -> class label mapping
- `metrics.json` – full metrics including classification report and confusion matrix
- `metrics_summary.csv` – condensed summary for your paper/report

## Notes

- Features: HOG on 128×128 grayscale + 3×8-bin HSV histograms (normalized) concatenated.
- Model: `LogisticRegression(solver='saga', multi_class='multinomial', class_weight='balanced', max_iter=3000)` inside a `Pipeline` with `StandardScaler`.
- For comparisons (Decision Tree, SVM), you can re-use the same `build_dataset` function and swap the sklearn classifier.
- For binary predictions, the `predict_logistic.py` script will print `healthy` or `unhealthy` based on the saved `label_map.json`.
