"""MLflow-tracked training pipeline for the ResNet50 classifier."""
import os
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import logging

from src.data.dataset import load_config, build_dataframe, split_data, validate_dataset
from src.data.preprocessing import get_classifier_generators
from src.models.classifier import build_classifier, compile_classifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def train(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"] + "_classifier")

    with mlflow.start_run(run_name="classifier_training"):
        mlflow.log_params({
            "backbone": config["classifier"]["backbone"],
            "learning_rate": config["classifier"]["learning_rate"],
            "batch_size": config["data"]["batch_size"],
            "epochs": config["classifier"]["epochs"],
            "dropout_rate": config["classifier"]["dropout_rate"],
            "image_size": config["data"]["image_size"],
        })

        df = build_dataframe(config["paths"]["raw_data"])
        validate_dataset(df)
        train_df, val_df, test_df = split_data(df, config)
        train_gen, val_gen, test_gen = get_classifier_generators(train_df, val_df, test_df, config)

        model = build_classifier(config)
        model = compile_classifier(model, config)

        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", verbose=1,
                          patience=config["classifier"]["patience_early_stop"], restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join(config["paths"]["models_dir"], "classifier-resnet-weights.keras"),
                            monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=config["classifier"]["lr_reduction_factor"],
                              patience=config["classifier"]["patience_reduce_lr"],
                              min_lr=config["classifier"]["min_lr"], verbose=1)
        ]

        history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.n // config["data"]["batch_size"],
            epochs=config["classifier"]["epochs"],
            validation_data=val_gen,
            validation_steps=val_gen.n // config["data"]["batch_size"],
            callbacks=callbacks
        )

        # Log metrics per epoch
        for epoch, (acc, loss, val_acc, val_loss) in enumerate(zip(
            history.history["accuracy"], history.history["loss"],
            history.history["val_accuracy"], history.history["val_loss"]
        )):
            mlflow.log_metrics({"train_acc": acc, "train_loss": loss,
                                 "val_acc": val_acc, "val_loss": val_loss}, step=epoch)

        # Evaluate on test set
        test_preds_prob = model.predict(test_gen, steps=test_gen.n // config["data"]["batch_size"])
        test_preds = np.argmax(test_preds_prob, axis=1)
        y_true = test_df["has_mask"].values[:len(test_preds)]
        report = classification_report(y_true, test_preds, output_dict=True)
        mlflow.log_metrics({
            "test_accuracy": report["accuracy"],
            "test_f1_macro": report["macro avg"]["f1-score"],
            "test_precision": report["macro avg"]["precision"],
            "test_recall": report["macro avg"]["recall"],
        })

        mlflow.keras.log_model(model, artifact_path="classifier_model",
                               registered_model_name="brain_tumor_classifier")
        logger.info("Classifier training complete and model registered in MLflow.")
        return model, history


if __name__ == "__main__":
    train()
