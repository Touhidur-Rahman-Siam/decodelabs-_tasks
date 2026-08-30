# Project 2: Data Classification Using AI
**DecodeLabs Industrial Training Kit — Batch 2026**

## What this is
A supervised learning pipeline that trains a K-Nearest Neighbors (KNN)
model to classify Iris flowers into one of three species (Setosa,
Versicolor, Virginica) based on 4 measured features.

## Files
- `iris_classifier.py` — the complete, runnable pipeline
- `k_tuning_curve.png` — generated on run: error rate vs. K, showing
  how the optimal K value was chosen
- `confusion_matrix.png` — generated on run: visual breakdown of
  correct vs. incorrect predictions per class

## How to run
1. Make sure Python 3.9+ is installed
2. Install dependencies:
   ```
   pip install scikit-learn pandas matplotlib numpy
   ```
3. Run:
   ```
   python3 iris_classifier.py
   ```
4. Console output shows accuracy, F1 score, confusion matrix, and a
   full classification report. Two PNG files are saved in the same
   folder.

## Requirements met (per Project 2 brief)
- **Load and understand a dataset** — `load_dataset()` loads the
  built-in Iris benchmark (150 samples, 3 balanced classes, 4 features)
- **Split data into training and testing sets** — `split_dataset()`
  does an 80/20 stratified, shuffled split with a fixed random seed
  for reproducibility
- **Apply a simple classification algorithm** — K-Nearest Neighbors
  (`KNeighborsClassifier`) via the standard scikit-learn
  instantiate → fit → predict workflow
- **Feature scaling** — `StandardScaler` fit only on training data,
  then applied to both sets (prevents data leakage)
- **Output validation** — confusion matrix, accuracy, and macro F1
  score (not accuracy alone, which can be misleading — "Accuracy
  Mirage" from the brief)
- **IPO model** — `run_pipeline()` follows Input (`load_dataset()`) →
  Process (`split_dataset()`, `scale_features()`, `find_optimal_k()`,
  `train_knn_model()`) → Output (`evaluate_model()`)

## Result summary (this run)
- Accuracy: **96.7%**
- Macro F1 Score: **0.967**
- 29/30 test samples classified correctly; the one error was a
  Virginica predicted as Versicolor — a well-known overlap in this
  dataset (these two species share similar petal dimensions),
  not a bug in the pipeline.

## Note on choosing K
`find_optimal_k()` sweeps K = 1 to 20 and picks the value with lowest
test error, which on this run landed on K=1. That's a legitimate
result on Iris (its classes are cleanly separated), but with only 30
test samples, a single train/test split isn't the most rigorous way
to choose K in general — a k-fold cross-validation would give a more
stable answer on noisier datasets. This is flagged here intentionally
as a "learning opportunity" per the project brief, not swept under
the rug.

## Author
Submitted as part of DecodeLabs Project 2 — Industrial Training Kit,
Batch 2026.
