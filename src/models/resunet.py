"""ResUNet segmentation model with skip connections."""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, Add,
    Conv2DTranspose, Input, concatenate, MaxPooling2D
)
import tensorflow.keras.backend as K
import logging

logger = logging.getLogger(__name__)


def residual_block(x, filters, kernel_size=3):
    shortcut = Conv2D(filters, 1, padding="same")(x)
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_resunet(config: dict) -> Model:
    img_size = config["data"]["image_size"]
    inputs = Input(shape=(*img_size, 3))

    # Encoder
    e1 = residual_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(e1)
    e2 = residual_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(e2)
    e3 = residual_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(e3)
    e4 = residual_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(e4)

    # Bridge
    b = residual_block(p4, 512)

    # Decoder
    d4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(b)
    d4 = concatenate([d4, e4])
    d4 = residual_block(d4, 256)

    d3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(d4)
    d3 = concatenate([d3, e3])
    d3 = residual_block(d3, 128)

    d2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(d3)
    d2 = concatenate([d2, e2])
    d2 = residual_block(d2, 64)

    d1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(d2)
    d1 = concatenate([d1, e1])
    d1 = residual_block(d1, 32)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(d1)
    model = Model(inputs, outputs, name="ResUNet_Segmentation")
    logger.info(f"Built ResUNet with {model.count_params():,} parameters")
    return model


def compile_resunet(model: Model, config: dict) -> Model:
    cfg = config["segmentation"]
    from src.losses import focal_tversky_loss, tversky_score
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"], epsilon=cfg["epsilon"])
    model.compile(optimizer=optimizer, loss=focal_tversky_loss, metrics=[tversky_score])
    return model
