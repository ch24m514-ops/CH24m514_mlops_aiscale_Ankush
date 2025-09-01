import json
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml import PipelineModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import numpy as np
import os


def evaluate_model():
    spark = SparkSession.builder.appName("EvaluateModel").getOrCreate()

    # Load test data
    test_data = spark.read.parquet("data/prepared/test.parquet")

    # Load preprocessor and classifier
    preprocessor = PipelineModel.load("models/preprocessor")
    model = GBTClassificationModel.load("models/classifier")

    # Apply preprocessing
    test_preprocessed = preprocessor.transform(test_data)

    # Make predictions
    predictions = model.transform(test_preprocessed)

    # Convert to pandas for sklearn metrics
    pdf = predictions.select("Survived", "prediction", "probability").toPandas()

    # Convert DenseVector -> float probability of class 1
    pdf["probability"] = pdf["probability"].apply(
        lambda v: float(v[1]) if hasattr(v, "__getitem__") else float(v)
    )

    y_true = pdf["Survived"].values
    y_pred = pdf["prediction"].values
    y_prob = pdf["probability"].values

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("✅ Model evaluation complete.")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1       : {f1:.4f}")
    print(f"   Roc_auc  : {roc_auc:.4f}")

    # Save metrics
    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("📊 Metrics saved to metrics/scores.json")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = [0, 1]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("📈 Confusion matrix saved to reports/confusion_matrix.png")

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC Curve)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("reports/roc_curve.png", bbox_inches="tight")
    plt.close()
    print("📉 ROC curve saved to reports/roc_curve.png")


if __name__ == "__main__":
    evaluate_model()
