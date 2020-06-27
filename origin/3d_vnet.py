from keras.layers import Input, Conv3D, BatchNormalization, Deconvolution3D, concatenate, PReLU, add, Lambda, LeakyReLU, ReLU
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
def vnet_3d(input_shape, n_classes=2, n_base_filters=16, kernel_size=5, depth=5, bn=True, deconv=True,
            lr=1e-5, decay=5e-4):
    inpt = Input(input_shape)
    x = inpt

    shortcuts = []
    # encoder
    for i in range(depth):   # [0,1,2,3,4]
        # resconv block
        x = conv_block(x, n_filters=n_base_filters*(2**i), kernel_size=kernel_size, depth=i, batch_normalization=bn)
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
            x = Lambda(resize_trilinear, arguments={'factors':(2,2,2)})(x)
            x = Conv3D(n_base_filters*(2**i), kernel_size=1, strides=1)(x)
        # concatenate
        x = concatenate([x,shortcuts[i]], axis=-1)
        # resconv block
        x = conv_block(x, n_filters=n_base_filters*(2**i)*2, kernel_size=kernel_size, depth=i, batch_normalization=bn)

    # head
    x = Conv3D(n_classes, kernel_size=1, strides=1, padding='same', activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(optimizer=Adam(lr), loss=dice_loss)

    return model


def conv_block(inpt, n_filters, kernel_size, depth, batch_normalization=True):
    n_blocks = min(depth+1, 3)
    x = inpt
    for i in range(n_blocks):
        x = Conv3D(n_filters, kernel_size, strides=1, padding='same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        else:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
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


def resize_trilinear(inputs, factors):
    num_batches, depth, height, width, num_channels = inputs._keras_shape
    dtype = inputs.dtype
    output_depth, output_height, output_width = [int(s * f) for s, f in zip([depth, height, width], factors)]

    # resize y-z
    squeeze_b_x = tf.reshape(inputs, [-1, height, width, num_channels])
    resize_b_x = tf.cast(tf.image.resize_bilinear(squeeze_b_x, [output_height, output_width], align_corners=True, half_pixel_centers=True), dtype=dtype)
    resume_b_x = tf.reshape(resize_b_x, [-1, depth, output_height, output_width, num_channels])

    # resize x
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    squeeze_b_z = tf.reshape(reoriented, [-1, output_height, depth, num_channels])
    resize_b_z = tf.cast(tf.image.resize_bilinear(squeeze_b_z, [output_height, output_depth], align_corners=True, half_pixel_centers=True), dtype=dtype)
    resume_b_z = tf.reshape(resize_b_z, [-1, output_width, output_height, output_depth, num_channels])

    # reorient back
    output = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output


if __name__ == '__main__':

    model = vnet_3d((128,128,128,1), n_classes=1, kernel_size=3, depth=5, bn=True, deconv=False)
    model.summary()




