"""Grad-CAM explainability for the ResNet50 brain tumor classifier."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model,
                          last_conv_layer: str = "conv5_block3_out") -> tuple:
    """Generate Grad-CAM heatmap for a given image and model."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()


def overlay_heatmap(heatmap: np.ndarray, original_img: np.ndarray,
                     alpha: float = 0.4) -> tuple:
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = (original_img * 255).astype(np.float32) if original_img.max() <= 1.0 else original_img.astype(np.float32)
    superimposed = heatmap_colored * alpha + img_uint8
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return heatmap_colored, superimposed


def load_and_preprocess(image_path: str, image_size: tuple = (256, 256)) -> tuple:
    """Load an image and return both the display version and model-ready array."""
    img_raw = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)
    img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    return img_resized, img_array


def run_gradcam(model: tf.keras.Model,
                test_df: pd.DataFrame,
                report_dir: str = "reports/",
                n_samples: int = 4,
                image_size: tuple = (256, 256),
                last_conv_layer: str = "conv5_block3_out") -> str:
    """
    Generate Grad-CAM visualizations for n_samples test images.

    Returns path to saved figure.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Pick balanced samples: half tumor, half no-tumor
    half = n_samples // 2
    tumor_samples    = test_df[test_df["has_mask"].astype(str) == "1"].head(half)
    no_tumor_samples = test_df[test_df["has_mask"].astype(str) == "0"].head(half)
    samples = pd.concat([tumor_samples, no_tumor_samples]).reset_index(drop=True)

    fig, axes = plt.subplots(len(samples), 3, figsize=(14, 4 * len(samples)))
    col_titles = ["Input MRI", "Grad-CAM Heatmap", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold")

    for i, (_, row) in enumerate(samples.iterrows()):
        img_display, img_array = load_and_preprocess(row["image_path"], image_size)
        heatmap, preds = make_gradcam_heatmap(img_array, model, last_conv_layer)
        heatmap_colored, superimposed = overlay_heatmap(heatmap, img_display.astype(np.float32) / 255.0)

        pred_class  = "Tumor" if np.argmax(preds[0]) == 1 else "No Tumor"
        confidence  = float(np.max(preds[0]))
        true_label  = "Tumor" if str(row["has_mask"]) == "1" else "No Tumor"
        correct     = "✅" if pred_class == true_label else "❌"

        axes[i][0].imshow(img_display)
        axes[i][0].set_ylabel(
            f"True: {true_label}\nPred: {pred_class} ({confidence:.2%}) {correct}",
            fontsize=9
        )
        axes[i][0].axis("off")

        axes[i][1].imshow(heatmap_colored)
        axes[i][1].axis("off")

        axes[i][2].imshow(superimposed)
        axes[i][2].axis("off")

    plt.suptitle("Grad-CAM Explainability — ResNet50 Brain Tumor Classifier",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(report_dir, "gradcam_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM saved to {save_path}")
    return save_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.data.dataset import load_config, build_dataframe, split_data

    config = load_config("configs/config.yaml")
    df = build_dataframe(config["paths"]["raw_data"])
    _, _, test_df = split_data(df, config)

    import mlflow.keras
    mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
    model = mlflow.keras.load_model("models:/brain_tumor_classifier/Production")

    save_path = run_gradcam(
        model=model,
        test_df=test_df,
        report_dir=config["paths"]["reports_dir"],
        n_samples=4
    )
    print(f"✅ Grad-CAM complete → {save_path}")
