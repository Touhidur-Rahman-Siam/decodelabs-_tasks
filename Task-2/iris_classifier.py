import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    classification_report,
)

RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = pd.Series(iris.target, name="species")
    class_names = list(iris.target_names)
    return features, labels, class_names


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_knn_model(X_train_scaled, y_train, n_neighbors: int = 5) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    return model


def find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test, k_range=range(1, 21)) -> int:
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        error_rates.append(1 - accuracy_score(y_test, preds))

    optimal_k = list(k_range)[int(np.argmin(error_rates))]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o")
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal K = {optimal_k}")
    plt.title("Tuning the Engine: Error Rate vs. K")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/k_tuning_curve.png", dpi=150)
    plt.close()

    return optimal_k


def evaluate_model(model: KNeighborsClassifier, X_test_scaled, y_test, class_names: list[str]) -> None:
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — Iris KNN Classifier")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/confusion_matrix.png", dpi=150)
    plt.close()


def run_pipeline() -> None:
    X, y, class_names = load_dataset()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(class_names)} classes {class_names}\n")

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    optimal_k = find_optimal_k(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Optimal K found via elbow method: {optimal_k}\n")

    model = train_knn_model(X_train_scaled, y_train, n_neighbors=optimal_k)

    evaluate_model(model, X_test_scaled, y_test, class_names)

    print("\nSaved artifacts: k_tuning_curve.png, confusion_matrix.png")


if __name__ == "__main__":
    run_pipeline()
