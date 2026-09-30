import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None
):
    """
    Calculate binary classification metrics.

    Class mapping:
        0 = REAL
        1 = AI-GENERATED
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )
    }

    if y_prob is not None:

        y_prob = np.asarray(y_prob)

        try:

            results["roc_auc"] = roc_auc_score(
                y_true,
                y_prob
            )

        except ValueError:

            results["roc_auc"] = None

    return results


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("EVALUATION METRICS TEST")
    print("=" * 60)

    # Dummy labels only for testing this module.
    y_true = [
        0, 0, 1, 1,
        0, 1, 0, 1
    ]

    y_pred = [
        0, 0, 1, 0,
        0, 1, 1, 1
    ]

    # Probability assigned to class 1:
    # AI-GENERATED
    y_prob = [
        0.10,
        0.20,
        0.90,
        0.40,
        0.15,
        0.80,
        0.65,
        0.85
    ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )

    print("\nClass mapping:")
    print("0 = REAL")
    print("1 = AI-GENERATED")

    print("\nEvaluation results:")

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{metrics['f1_score']:.4f}"
    )

    if metrics.get("roc_auc") is not None:

        print(
            f"ROC-AUC:   "
            f"{metrics['roc_auc']:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )

    print("\nConfusion matrix format:")

    print(
        "[[True REAL,  REAL predicted as AI],"
    )

    print(
        " [AI predicted as REAL, True AI]]"
    )

    print("\n" + "=" * 60)
    print("EVALUATION METRICS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()