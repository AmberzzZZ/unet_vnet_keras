from keras.layers import Input, MaxPooling3D, Conv3D, BatchNormalization, Activation,  \
                          Deconvolution3D, UpSampling3D, concatenate
from keras.optimizers import Adam
from keras.models import Model
from loss import *


def unet_3d(input_shape, n_labels=1, lr=1e-5, depth=4, n_base_filters=32, bn=False, deconv=False):
    inpt = Input(input_shape)
    x = inpt

    skips = []
    # encoder
    for i in range(depth):   # [0,1,2,3]
        x = conv_block(x, n_filters=n_base_filters*(2**i), batch_normalization=bn)
        x = conv_block(x, n_filters=n_base_filters*(2**i)*2, batch_normalization=bn)
        skips.append(x)
        if i < depth - 1:
            x = MaxPooling3D(pool_size=2)(x)

    # decoder
    for i in range(depth-2, -1, -1):     # [2,1,0]
        x = up_conv(x, n_filters=x._keras_shape[-1], deconvolution=deconv)
        x = concatenate([x, skips[i]], axis=-1)
        x = conv_block(x, n_filters=skips[i]._keras_shape[-1], batch_normalization=bn)
        x = conv_block(x, n_filters=skips[i]._keras_shape[-1], batch_normalization=bn)

    # head
    x = Conv3D(n_labels, 1)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inpt, outputs=x)

    model.compile(optimizer=Adam(lr), loss=dice_loss)
    return model


def conv_block(input, n_filters, kernel_size=3, activation='relu', padding='same', strides=1,
                batch_normalization=False, instance_normalization=False):
    x = Conv3D(n_filters, kernel_size, padding=padding, strides=strides)(input)
    if batch_normalization:
        x = BatchNormalization(axis=1)(x)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        x = InstanceNormalization(axis=1)(x)
    return Activation(activation)(x)


def up_conv(input, n_filters, pool_size=2, kernel_size=2, strides=2, deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides)(input)
    else:
        return UpSampling3D(size=pool_size)(input)


if __name__ == '__main__':

    model = unet_3d((128,128,128,3), n_labels=4, lr=1e-5, bn=True, deconv=True)
    model.summary()




