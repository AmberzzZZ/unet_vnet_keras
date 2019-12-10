from keras.layers import Input,BatchNormalization,Conv2D,Dropout,PReLU,Conv2DTranspose,  \
                         add,concatenate,Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from keras.losses import binary_crossentropy,mean_squared_error
import os
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(K.log(pt_1)) - K.mean(K.log(1.-pt_0))
    return score


def reweighting_bce(y_true, y_pred):
    alpha = 0.999
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(alpha * K.log(pt_1)) - K.mean((1-alpha) * K.log(1.-pt_0))
    # score = -K.mean(alpha * K.log(pt_1), axis=(0,1,2)) - K.mean((1-alpha) * K.log(1.-pt_0), axis=(0,1,2))
    # beta = tf.constant([0.02, 0.49, 0.49])
    # score = K.sum(beta * score)
    return score


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = 0.25
    # score = alpha * y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred) +            # this works when y_true==1
    #         (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma) * K.log(1 - y_pred)  # this works when y_true==0
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # avoid nan
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)
    score = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) -  \
            K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return score


def mix_loss(y_true, y_pred):
    alpha = 0.01
    focal_dice = alpha * focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    mse_dice = alpha * mean_squared_error(y_true, y_pred) + dice_loss(y_true, y_pred)
    bce_dice = alpha * reweighting_bce(y_true, y_pred) + dice_loss(y_true, y_pred)
    return bce_dice


def Conv_block(x, n_filters, kernel_size, strides=1, padding='same',
               activation=None, kernel_initializer='he_normal', transpose=False):
    if transpose:
        x = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=activation, kernel_initializer=kernel_initializer)(x)
    else:
        x = Conv2D(n_filters, kernel_size=kernel_size, padding=padding, strides=strides,
                   activation=None, kernel_initializer='he_normal')(x)
    x = PReLU()(x)

    return x


# for compression
def resBlock(inpt, stage):
    # conv
    x = inpt
    for _ in range(min(stage, 3)):     # [1,2,3,4]
        x = Conv_block(x, n_filters=16*(2**(stage-1)), kernel_size=5)
    feature = add([x, inpt])

    # downsampling
    if stage < 5:
        x = Conv_block(feature, n_filters=16*(2**stage), kernel_size=2, strides=2)
        x = PReLU()(x)

    return x, feature


def pad(args):
    x, inpt = args
    inpt = tf.pad(inpt,[[0,0],[0,0],[0,0],[0,x.shape[3]-inpt.shape[3]]])
    return inpt


# for decompression
def up_resBlock(inpt, shortcut, stage):
    # deconv
    x = inpt
    x = Conv_block(x, n_filters=16*(2**(stage-1)), kernel_size=2,
                   strides=2, padding='valid', transpose=True)
    x = PReLU()(x)
    inpt = x     # save for residual
    # concatenate
    x = concatenate([shortcut, x], axis=-1)
    # conv residual
    for _ in range(min(stage, 3)):    # [4,3,2,1]
        x = Conv_block(x, n_filters=16*(2**(stage)), kernel_size=5)
    if x.shape[3] != inpt.shape[3]:
        # zeros padding to the same
        inpt = Lambda(pad)([x, inpt])
    x = add([x, inpt])

    return x


def vnet(input_size=(128,128,1),num_classes=1,stage_num=5,thresh=0.5):
    inpt = Input(input_size)

    x = inpt
    # compression
    features=[]
    for s in range(1, stage_num+1):   # [1,2,3,4,5]
        x, feature = resBlock(x, s)
        features.append(feature)
        # print("compression stage %d:  output:   " % s, feature)

    # decompression
    for d in range(stage_num-1, 0, -1):     # [4,3,2,1]
        x = up_resBlock(x, features[d-1], d)
        # print("decompression stage %d:  output:   " % d, x)

    # last layer
    x = Conv2D(filters=num_classes, kernel_size=1, padding='same',
               activation='sigmoid', kernel_initializer='he_normal')(x)

    model = Model(inputs=inpt,outputs=x)

    sgd = SGD(lr=1e-3, momentum=0.9, decay=0., nesterov=True)
    model.compile(optimizer=sgd, loss=mix_loss, metrics=[dice_coef, dice_loss, reweighting_bce])

    return model


if __name__ == '__main__':
    model = vnet(input_size=(128,128,1), num_classes=1, stage_num=5, thresh=0.5)
    model.summary()







