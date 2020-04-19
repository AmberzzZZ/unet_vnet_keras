from keras.layers import Input, Conv3D, BatchNormalization, Deconvolution3D, concatenate, add, Lambda, LeakyReLU, ZeroPadding3D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.models import Model
from loss import *
import tensorflow as tf
from DepthwiseConv3D import DepthwiseConv3D


###### model ######
def vnet_3d_depthwise(input_shape, n_classes=2, n_base_filters=16, depth=5, kernel_size=5, bn=True, deconv=False,
                      lr=1e-5):
    inpt = Input(input_shape)

    x = Conv3D(n_base_filters, 3, strides=1, padding='same')(inpt)
    shortcuts = [x]
    # encoder
    for i in range(depth-1):   # [0,1,2,3]
        # down conv
        x = Conv3D(n_base_filters*(2**i)*2, kernel_size=2, strides=2)(x)
        # resconv block
        x = conv_block(x, n_filters=n_base_filters*(2**i), kernel_size=kernel_size, depth=i, batch_normalization=bn)
        shortcuts.append(x)

    # decoder
    for i in range(depth-2, -1, -1):     # [3,2,1,0]
        # up samp
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
        x = ZeroPadding3D(padding=(1, 1, 1))(x)
        x = DepthwiseConv3D(kernel_size, depth_multiplier=1, padding='valid')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        else:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
    # residual
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

    model = vnet_3d_depthwise((128,128,128,1), n_classes=1, depth=5, kernel_size=3, bn=True, deconv=False)
    model.summary()



