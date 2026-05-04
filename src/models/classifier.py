"""ResNet50-based binary classifier for tumor detection."""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (AveragePooling2D, Dense, Dropout, Flatten, Input)
import logging

logger = logging.getLogger(__name__)


def build_classifier(config: dict) -> Model:
    cfg = config["classifier"]
    img_size = config["data"]["image_size"]
    input_shape = (*img_size, 3)

    base_model = ResNet50(weights=cfg["weights"], include_top=False, input_tensor=Input(shape=input_shape))
    if cfg["freeze_base"]:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten(name="flatten")(x)
    x = Dense(cfg["dense_units"][0], activation="relu")(x)
    x = Dropout(cfg["dropout_rate"])(x)
    x = Dense(cfg["dense_units"][1], activation="relu")(x)
    x = Dropout(cfg["dropout_rate"])(x)
    outputs = Dense(cfg["num_classes"], activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Classifier")
    logger.info(f"Built classifier with {model.count_params():,} parameters")
    return model


def compile_classifier(model: Model, config: dict) -> Model:
    cfg = config["classifier"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
