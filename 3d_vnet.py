from keras.layers import Input, MaxPooling3D, Conv3D, BatchNormalization,  \
                          Deconvolution3D, UpSampling3D, concatenate, PReLU, add, Lambda
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.models import Model
from loss import *
import tensorflow as tf


###### custom loss ######
def weighted_dice_loss(y_true, y_pred):
    pos_mask = y_true
    neg_mask = 1 - y_true
    dice_p = dice_loss(y_true*pos_mask, y_pred*pos_mask)
    dice_n = dice_loss(y_true*neg_mask, y_pred*neg_mask)
    return dice_p*0.8 + dice_n*0.2


def mixed_loss(y_true, y_pred):
    # weighted dice loss
    dice = weighted_dice_loss(y_true, y_pred)
    # bce
    bce = reweighting_bce(y_true, y_pred)
    return dice + bce


###### custom metric ######
metric_lst = [weighted_dice_loss, reweighting_bce]


###### model ######
def vnet_3d(input_shape, n_classes=2, n_base_filters=16, depth=5, bn=True, deconv=True,
            lr=1e-5, decay=5e-4):
    inpt = Input(input_shape)
    x = inpt

    shortcuts = []
    # encoder
    for i in range(depth):   # [0,1,2,3,4]
        # resconv block
        x = conv_block(x, n_filters=n_base_filters*(2**i), depth=i, batch_normalization=bn)
        shortcuts.append(x)
        # down conv
        if i < depth-1:
            x = Conv3D(n_base_filters*(2**i)*2, kernel_size=2, strides=2)(x)

    # decoder
    for i in range(depth-2, -1, -1):     # [3,2,1,0]
        # up conv
        if deconv:
            x = Deconvolution3D(n_base_filters*(2**i), kernel_size=2, strides=2)(x)
        else:
            x = UpSampling3D(size=2)(x)
            x = Conv3D(n_base_filters*(2**i), kernel_size=1, strides=1)(x)
        # concatenate
        print(x)
        print(shortcuts[i])
        x = concatenate([x,shortcuts[i]], axis=-1)
        # resconv block
        x = conv_block(x, n_filters=n_base_filters*(2**i)*2, depth=i, batch_normalization=bn)

    # head
    x = Conv3D(n_classes, kernel_size=1, strides=1, padding='same', activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(optimizer=Adam(lr), loss=dice_loss)

    return model


def conv_block(inpt, n_filters, depth, batch_normalization=True):
    n_blocks = min(depth+1, 3)
    x = inpt
    for i in range(n_blocks):
        x = Conv3D(n_filters, kernel_size=5, strides=1, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        else:
            x = InstanceNormalization()(x)
        x = PReLU()(x)
    x = Lambda(element_add)([x, inpt])
    return x


def element_add(args):
    x, inpt = args
    gap = x._keras_shape[-1] - inpt._keras_shape[-1]
    if gap > 0:
        inpt = tf.pad(inpt, [[0,0], [0,0], [0,0], [0,0], [0,gap]])
    else:
        x = tf.pad(x, [[0,0], [0,0], [0,0], [0,0], [0,gap]])
    return add([x, inpt])


if __name__ == '__main__':

    model = vnet_3d((128,128,128,1), n_classes=1, depth=4, bn=True, deconv=False)
    model.summary()




