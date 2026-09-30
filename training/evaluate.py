import torch
import numpy as np
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

from training.dataloader import create_dataloader
from training.model_cnn_transformer import CNNTransformer
from training.config import DEVICE


MODEL_PATH = Path(
    "saved_models/cnn_transformer_final.pth"
)

RESULTS_DIR = Path(
    "outputs/reports"
)

PLOTS_DIR = Path(
    "outputs/plots"
)


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE")
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    print("\nLoading test dataset...")

    test_loader = create_dataloader(
        "test",
        shuffle=False
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------

    model = CNNTransformer(
        num_classes=2
    ).to(DEVICE)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # RUN TEST
    # --------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\nRunning test evaluation...\n")

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    # --------------------------------------------------
    # NUMPY ARRAYS
    # --------------------------------------------------

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    y_probability = np.array(
        all_probabilities
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    else:

        roc_auc = None

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1-Score  : {f1 * 100:.2f}%"
    )

    if roc_auc is not None:

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        "\n                 Predicted"
    )

    print(
        "              REAL   AI"
    )

    print(
        f"Actual REAL   {cm[0][0]:4d}  {cm[0][1]:4d}"
    )

    print(
        f"Actual AI     {cm[1][0]:4d}  {cm[1][1]:4d}"
    )

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "REAL",
            "AI-GENERATED"
        ],
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    # --------------------------------------------------
    # SAVE CONFUSION MATRIX IMAGE
    # --------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "AI Voice vs Real Voice - Confusion Matrix"
    )

    plt.colorbar()

    class_names = [
        "REAL",
        "AI-GENERATED"
    ]

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    # Write values inside cells
    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    confusion_matrix_path = (
        PLOTS_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nConfusion matrix saved:"
    )

    print(
        confusion_matrix_path
    )

    # --------------------------------------------------
    # SAVE METRICS JSON
    # --------------------------------------------------

    metrics = {
        "model": "CNNTransformer",
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "confusion_matrix": cm.tolist()
    }

    metrics_path = (
        RESULTS_DIR /
        "test_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "Metrics saved:"
    )

    print(
        metrics_path
    )

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()