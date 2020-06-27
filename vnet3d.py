from keras.layers import Input, Conv3D, Conv3DTranspose, Lambda, concatenate, MaxPooling3D, add, \
                         Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from loss import *
import tensorflow as tf


##### custom loss ######
def mixed_loss(y_true, y_pred):
    # dice
    dice = 1 - dice_coef(y_true, y_pred)
    # bce
    bce = binary_crossentropy(y_true, y_pred)
    return dice + bce


##### custom metric #####
metric_lst = [dice_coef, binary_crossentropy]


def vnet_3d(input_tensor=None, input_shape=(64,64,64,1), n_classes=1, depth=4,
            lr=1e-5, decay=5e-4):
    if input_tensor:
        inpt = input_tensor
    else:
        inpt = Input(input_shape)

    start_filters = 32 if depth==4 else 16
    # encoder
    x, feature1 = conv_block(inpt, n_filters=start_filters*1, n_blocks=1, downSamp=True)
    x, feature2 = conv_block(x, n_filters=start_filters*2, n_blocks=3, downSamp=True, Inception=True)
    x, feature3 = conv_block(x, n_filters=start_filters*4, n_blocks=4, downSamp=True, Inception=True)
    if depth==5:
        x, feature4 = conv_block(x, n_filters=start_filters*8, n_blocks=6, downSamp=True, Inception=True)
    x, _ = conv_block(x, n_filters=256, n_blocks=3)

    # decoder
    if depth==5:
        x, _ = conv_block(x, n_filters=start_filters*8, n_blocks=3, upSamp=True, shortcut=feature4)
    x, _ = conv_block(x, n_filters=start_filters*4, n_blocks=3, upSamp=True, shortcut=feature3)
    x, _ = conv_block(x, n_filters=start_filters*2, n_blocks=2, upSamp=True, shortcut=feature2)
    x, _ = conv_block(x, n_filters=start_filters*1, n_blocks=1, upSamp=True, shortcut=feature1)

    # head
    x = Conv3D(n_classes, kernel_size=1, strides=1, padding='same', activation='sigmoid')(x)

    model = Model(inputs=inpt, outputs=x)

    model.compile(Adam(lr=lr, decay=decay), loss=mixed_loss, metrics=metric_lst)

    return model


def conv_block(x, n_filters, n_blocks, downSamp=False, Inception=False, upSamp=False, shortcut=None):
    inpt = x

    if upSamp:
        # x = Conv3DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
        x = Lambda(resize_trilinear, arguments={'factors':(2,2,2)})(x)
        x = Conv3D(n_filters, kernel_size=1, strides=1, padding='same')(x)
        inpt = x
        x = concatenate([x, shortcut], axis=-1)

    # x = vnet_block(x, inpt, n_filters, n_blocks)
    x = res_block(x, n_filters, n_blocks)
    feature = x

    if downSamp:
        if Inception:
            x = InceptionDownSamp(x)
        else:
            x = Conv_BN(x, n_filters*2, strides=2, activation='relu')
            # x = MaxPooling3D(pool_size=2, strides=2)(x)

    return x, feature


# successive 3x3convs + id path
def vnet_block(x, inpt, n_filters, n_blocks):
    for i in range(n_blocks):
        if i==n_blocks-1:
            x = Conv_BN(x, n_filters)
        else:
            x = Conv_BN(x, n_filters, activation='relu')

    if x._keras_shape[-1]==inpt._keras_shape[-1]:
        x = add([x, inpt])              # id add
    else:
        inpt = Conv_BN(inpt, n_filters, kernel_size=1, activation=None)
        x = add([x, inpt])              # conv add
    x = Activation('relu')(x)
    return x


# successive resconvs
def res_block(x, n_filters, n_blocks):
    for i in range(n_blocks):
        x = Res_Conv_BN(x, n_filters)
    return x


# 1x1 + 3x3 + 1x1
def Res_Conv_BN(x, n_filters):
    inpt = x
    x = Conv_BN(x, n_filters//4, kernel_size=1, strides=1, activation='relu')
    x = Conv_BN(x, n_filters//4, kernel_size=3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, kernel_size=1, strides=1, activation=None)

    if x._keras_shape[-1]==inpt._keras_shape[-1]:
        x = add([x, inpt])              # id add
    else:
        inpt = Conv_BN(inpt, n_filters, kernel_size=1, activation=None)
        x = add([x, inpt])              # conv add
    x = Activation('relu')(x)
    return x


# conv + bn + activation
def Conv_BN(x, n_filters, kernel_size=3, strides=1, activation=None):
    x = Conv3D(n_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)
    return x


def InceptionDownSamp(x):
    in_channels = x._keras_shape[-1]
    x1 = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)
    x2 = Conv_BN(x, in_channels//2, kernel_size=3, strides=2, activation='relu')
    x3 = Conv_BN(x, in_channels//4, kernel_size=1, strides=1, activation='relu')
    x3 = Conv_BN(x3, in_channels//4, kernel_size=3, strides=1, activation='relu')
    x3 = Conv_BN(x3, in_channels//2, kernel_size=3, strides=2, activation='relu')
    x = concatenate([x1,x2,x3], axis=-1)
    return x


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

    model = vnet_3d(input_shape=(64,128,128,2), n_classes=1, depth=4)
    model.summary()























