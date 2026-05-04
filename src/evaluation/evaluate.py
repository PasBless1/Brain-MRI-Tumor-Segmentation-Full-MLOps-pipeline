"""Evaluation metrics: Dice score, IoU, confusion matrix, and visual reports."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import mlflow
import logging
import os

logger = logging.getLogger(__name__)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred.flatten() > 0.5).astype(float)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred.flatten() > 0.5).astype(float)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate_classifier(model, test_gen, test_df, report_dir: str = "reports/"):
    os.makedirs(report_dir, exist_ok=True)
    preds_prob = model.predict(test_gen)
    preds = np.argmax(preds_prob, axis=1)
    y_true = test_df["has_mask"].values[:len(preds)]

    report = classification_report(y_true, preds, target_names=["no_tumor", "tumor"])
    with open(os.path.join(report_dir, "classifier_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Tumor", "Tumor"]).plot(ax=ax)
    plt.title("Classifier Confusion Matrix")
    plt.savefig(os.path.join(report_dir, "classifier_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    auc = roc_auc_score(y_true, preds_prob[:, 1])
    fpr, tpr, _ = roc_curve(y_true, preds_prob[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    plt.savefig(os.path.join(report_dir, "classifier_roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(os.path.join(report_dir, "classifier_confusion_matrix.png"))
    mlflow.log_artifact(os.path.join(report_dir, "classifier_roc_curve.png"))
    mlflow.log_metric("test_auc", auc)
    logger.info(f"Classifier evaluation complete. AUC={auc:.3f}")
    return {"auc": auc}


def evaluate_segmentation(model, test_gen) -> dict:
    dice_scores, iou_scores = [], []
    for images, masks in test_gen:
        preds = model.predict(images, verbose=0)
        for gt, pr in zip(masks, preds):
            dice_scores.append(dice_score(gt, pr))
            iou_scores.append(iou_score(gt, pr))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    mlflow.log_metrics({"test_avg_dice": avg_dice, "test_avg_iou": avg_iou})
    logger.info(f"Segmentation evaluation: Avg Dice={avg_dice:.4f}, Avg IoU={avg_iou:.4f}")
    return {"avg_dice": avg_dice, "avg_iou": avg_iou}
