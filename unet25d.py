# 2.5d unet
from keras.layers import Input, Conv2D, Conv3D, BatchNormalization, ReLU, PReLU, Activation, \
                         MaxPool2D, MaxPool3D, concatenate, Lambda, multiply, add
from keras.models import Model
import tensorflow as tf
import keras.backend as K


def unet25d(input_shape=(512,512,5), n_classes=1):

    inpt = Input(input_shape)

    features = []

    # L1
    x = convBlock(inpt, 16, dim2=True)      # (?, 512, 512, 5, 16)
    features.append(x)
    x = MaxPool3D(pool_size=(2, 2, 1), padding='same')(x)  # (?, 256, 256, 5, 16)

    # L2
    x = convBlock(x, 32, dim2=True)       # (?, 256, 256, 5, 32)
    features.append(x)
    x = MaxPool3D(pool_size=(2, 2, 1), padding='same')(x)       # (?, 128, 128, 5, 32)

    # L3
    x = convBlock(x, 64, padding='same')    # (?, 128, 128, 5, 64)
    features.append(x)
    x = MaxPool3D(pool_size=3, strides=2, padding='same')(x)  # (?, 64, 64, 3, 64)

    # L4
    x = convBlock(x, 128, padding='same')     # (?, 64, 64, 3, 128)
    features.append(x)
    x = MaxPool3D(pool_size=3, strides=2, padding='same')(x)    # (?, 32, 32, 2, 128)

    # L5
    amp5, x = attentionModule(x)
    x = convBlock(x, 256)              # (?, 32, 32, 2, 256)

    # d-L4
    x = Lambda(resize_trilinear, arguments={'factors':(2,2,2)})(x)
    x = Conv3D(128, 3, strides=1, padding='same')(x)
    f4 = Lambda(tf.pad, arguments={'paddings': [[0,0],[0,0],[0,0],[0,1],[0,0]]})(features[3])
    x = concatenate([x, f4], axis=-1)
    amp4, x = attentionModule(x)
    x = convBlock(x, 128)
    x = Lambda(lambda x: x[:,:,:,:3,:], name='slice_1')(x)               # (?, 64, 64, 3, 128)

    # d-L3
    x = Lambda(resize_trilinear, arguments={'factors':(2,2,2)})(x)
    x = Conv3D(64, 3, strides=1, padding='same')(x)
    f3 = Lambda(tf.pad, arguments={'paddings': [[0,0],[0,0],[0,0],[0,1],[0,0]]})(features[2])
    x = concatenate([x, f3], axis=-1)
    amp3, x = attentionModule(x)
    x = convBlock(x, 64)
    x = Lambda(lambda x: x[:,:,:,:5,:], name='slice_2')(x)             # # (?, 128, 128, 5, 128)

    # d-L2
    x = Lambda(resize_trilinear, arguments={'factors':(2,2,1)})(x)
    x = Conv3D(32, (3,3,1), strides=1, padding='same')(x)
    x = concatenate([x, features[1]], axis=-1)
    amp2, x = attentionModule(x, dim2=True)
    x = convBlock(x, 32, dim2=True)

    # d-L1
    x = Lambda(resize_trilinear, arguments={'factors':(2,2,1)})(x)
    x = Conv3D(16, (3,3,1), strides=1, padding='same')(x)
    x = concatenate([x, features[0]], axis=-1)
    amp1, x = attentionModule(x, dim2=True)
    x = convBlock(x, 16, dim2=True)

    # head
    x = Conv3D(n_classes, (3,3,1), strides=1, padding='same', activation='sigmoid')(x)

    model = Model(inpt, x)

    return model


def convBlock(x, n_filters, n_convs=2, dim2=False, padding='same'):
    for i in range(n_convs):
        if dim2:
            x = Conv3D(n_filters, (3,3,1), strides=1, padding=padding)(x)
        else:
            x = Conv3D(n_filters, 3, strides=1, padding=padding)(x)
        x = BatchNormalization(axis=-1)(x)
        # x = PReLU()(x)
        x = ReLU()(x)
    return x


def attentionModule(x, dim2=False):
    inpt = x
    channels = x._keras_shape[-1]
    if dim2:
        x = Conv3D(channels//2, (3,3,1), strides=1, padding='same')(x)
    else:
        x = Conv3D(channels//2, 3, strides=1, padding='same')(x)
    x = ReLU()(x)
    if dim2:
        x = Conv3D(1, (3,3,1), strides=1, padding='same')(x)
    else:
        x = Conv3D(1, 3, strides=1, padding='same')(x)
    atten_map = Activation('sigmoid')(x)
    x = multiply([inpt, atten_map])
    x = add([inpt, x])
    return atten_map, x


def resize_trilinear(inputs, factors):
    num_batches, depth, height, width, num_channels = K.int_shape(inputs)
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

    model = unet25d(input_shape=(512,512,5,1), n_classes=1)
    model.summary()


