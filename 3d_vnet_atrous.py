from keras.layers import Input, Conv3D, BatchNormalization,  \
                          Deconvolution3D, UpSampling3D, concatenate, PReLU, add, Lambda, LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.models import Model
from loss import *
import tensorflow as tf



###### model ######
def vnet_3d_atrous(input_shape, n_classes=2, n_base_filters=16, depth=5, bn=True, deconv=False,
                    lr=1e-5, decay=5e-4):
    inpt = Input(input_shape)

    # encoder
    x1 = down_block(inpt, n_filters=n_base_filters, kernel_size=7, strides=1, n_blocks=1)
    x2 = down_block(x1, n_filters=n_base_filters*2, kernel_size=3, strides=2, n_blocks=2)
    x3 = down_block(x2, n_filters=n_base_filters*4, kernel_size=3, strides=2, n_blocks=2)
    x4 = down_block(x3, n_filters=n_base_filters*8, kernel_size=3, strides=2, n_blocks=3)
    x5 = down_block(x4, n_filters=n_base_filters*16, kernel_size=3, strides=2, n_blocks=4)

    # decoder
    x6 = up_block(x5, x4, n_filters=n_base_filters*16, n_blocks=3)
    x7 = up_block(x6, x3, n_filters=n_base_filters*8, n_blocks=2)
    x8 = up_block(x7, x2, n_filters=n_base_filters*4, n_blocks=2)
    x9 = up_block(x8, x1, n_filters=n_base_filters*2, n_blocks=1)

    # head
    x = Conv3D(n_classes, kernel_size=1, strides=1, padding='same', activation='softmax')(x9)

    model = Model(inputs=inpt, outputs=x)
    model.compile(optimizer=Adam(lr), loss=dice_loss)

    return model


def down_block(inpt, n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, n_blocks=1):
    # downSamp
    inpt = Conv3D(n_filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(inpt)
    # conv block
    x = inpt
    for i in range(n_blocks):
        x = conv_bn(x, n_filters, 3, strides=1, padding='same')
        x = conv_bn(x, n_filters, 1)
    return add([inpt,x])


def up_block(inpt, shortcut, n_filters, n_blocks):
    # upSamp
    inpt = Lambda(resize_trilinear, arguments={'factors':(2,2,2)})(inpt)
    inpt = Conv3D(n_filters//2, kernel_size=1, strides=1)(inpt)
    # concate
    inpt = concatenate([inpt, shortcut], axis=-1)
    # conv block
    x = inpt
    for i in range(n_blocks):
        x = conv_bn(x, n_filters, 3, strides=1, padding='same')
        x = conv_bn(x, n_filters, 1)
    return add([inpt, x])


def conv_bn(inpt, n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, norm='bn', activation='leaky'):
    x = Conv3D(n_filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate)(inpt)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
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

    model = vnet_3d_atrous((128,128,128,1), n_classes=1, depth=4, bn=True, deconv=False)
    model.summary()



