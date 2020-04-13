import keras.backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred = tf.where(y_pred>0.3, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_n(y_true, y_pred, channel):
    y_t = y_true[..., channel]
    y_p = y_pred[..., channel]
    return dice_coef(y_t, y_p)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = 0.99     # 0.25 in paper
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # avoid nan
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) -  \
            K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return score


def focal_loss_n(y_true, y_pred, channel):
    y_t = y_true[..., channel]
    y_p = y_pred[..., channel]
    return focal_loss(y_t, y_p)


def bce(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(K.log(pt_1)) - K.mean(K.log(1.-pt_0))
    return score


def reweighting_bce(y_true, y_pred):
    alpha = 0.99
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(alpha * K.log(pt_1)) - K.mean((1-alpha) * K.log(1.-pt_0))
    return score


def reweighting_bce_n(y_true, y_pred, channel):
    y_t = y_true[..., channel]
    y_p = y_pred[..., channel]
    return reweighting_bce(y_t, y_p)


def get_border(pool_size, y_pred):
    negative = 1 - y_pred
    positive = y_pred
    positive = K.pool2d(positive, pool_size=pool_size, padding="same")
    negative = K.pool2d(negative, pool_size=pool_size, padding="same")
    border = positive * negative
    return border


def border_dice_coef(y_true, y_pred):
    return dice_coef(get_border((3,3),y_true), get_border((3,3), y_pred))


def border_dice_n(y_true, y_pred, channel):
    y_t = y_true[..., channel:channel+1]
    y_p = y_pred[..., channel:channel+1]
    return border_dice_coef(y_t, y_p)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def bilinear(shape, dtype='float32'):
    in_channels, out_channels, kernel_size, kernel_size = shape
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(shape, dtype=dtype)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)



