"""Custom loss functions: Focal Tversky and Tversky Score."""
import tensorflow as tf
import tensorflow.keras.backend as K


def tversky_score(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1.0):
    y_true_f = K.flatten(K.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky_score(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky_score(y_true, y_pred)
    return K.pow((1 - tv), gamma)
