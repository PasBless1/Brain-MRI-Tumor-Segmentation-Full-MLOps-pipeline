"""MLflow-tracked training pipeline for the ResUNet segmentation model."""
import os
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

from src.data.dataset import load_config, build_dataframe, split_data, validate_dataset
from src.models.resunet import build_resunet, compile_resunet
from src.losses import focal_tversky_loss, tversky_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, config, augment=False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.augment = augment
        self.image_size = tuple(config["data"]["image_size"])
        self.batch_size = config["data"]["batch_size"]

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        import cv2
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []
        for _, row in batch.iterrows():
            img = cv2.imread(row["image_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size).astype(np.float32) / 255.0
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size).astype(np.float32) / 255.0
            mask = np.expand_dims(mask, -1)
            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)


def train(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"] + "_segmentation")

    with mlflow.start_run(run_name="resunet_training"):
        mlflow.log_params({
            "model": config["segmentation"]["model"],
            "learning_rate": config["segmentation"]["learning_rate"],
            "epochs": config["segmentation"]["epochs"],
            "batch_size": config["data"]["batch_size"],
            "image_size": config["data"]["image_size"],
            "tversky_alpha": config["segmentation"]["tversky_alpha"],
            "tversky_gamma": config["segmentation"]["tversky_gamma"],
        })

        df = build_dataframe(config["paths"]["raw_data"])
        sick_df = df[df["has_mask"] == 1].copy()
        _, _, test_full = split_data(df, config)
        train_df, val_df, _ = split_data(sick_df, config)

        train_gen = SegmentationDataGenerator(train_df, config, augment=True)
        val_gen = SegmentationDataGenerator(val_df, config)

        model = build_resunet(config)
        model = compile_resunet(model, config)

        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=config["segmentation"]["patience_early_stop"],
                          verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(config["paths"]["models_dir"], "ResUNet-weights.keras"),
                            monitor="val_loss", save_best_only=True, verbose=1)
        ]

        history = model.fit(
            train_gen, epochs=config["segmentation"]["epochs"],
            validation_data=val_gen, callbacks=callbacks
        )

        for epoch, (loss, tv, val_loss, val_tv) in enumerate(zip(
            history.history["loss"], history.history["tversky_score"],
            history.history["val_loss"], history.history["val_tversky_score"]
        )):
            mlflow.log_metrics({"train_loss": loss, "train_tversky": tv,
                                 "val_loss": val_loss, "val_tversky": val_tv}, step=epoch)

        mlflow.keras.log_model(model, artifact_path="resunet_model",
                               registered_model_name="brain_tumor_segmentor")
        logger.info("Segmentation model training complete and registered in MLflow.")
        return model, history


if __name__ == "__main__":
    train()
