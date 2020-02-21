from keras.models import Model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, MaxPool2D, Activation, Concatenate
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from functools import wraps, reduce


####### darknet52 ##########
def Darknet52(input_tensor=None, input_shape=(256,256,1), weights=None):

    if input_tensor is None:
        input_tensor = Input(input_shape)
    x = DarknetConv2D_BN_Leaky(32, (3,3))(input_tensor)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    model = Model(input_tensor, x)
    return model


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


####### origin_unet ##########
def OrigUnet(input_tensor=None, input_shape=(256,256,1), stage=5):

    if input_tensor is None:
        input_tensor = Input(input_shape)

    # compression
    x = input_tensor
    for i in range(stage):    # [0,1,2,3,4]
        x, feature = comp_block(x, i)
        # print(feature)

    model = Model(input_tensor, x)

    return model


def comp_block(x, stage, res=False):   # [0,1,2,3,4]
    x = conv_block(x, 64*(2**stage))
    feature = x
    if stage < 4:
        x = MaxPool2D()(x)
    return x, feature


def conv_block(x, n_filters, kernel_size=3, strides=1, padding='same', activation='relu', res=False):
    inpt = x
    x = Conv2D(n_filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(n_filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return Concatenate()([inpt, x]) if res else x


if __name__ == '__main__':

    # model = Darknet52(input_tensor=Input((512, 512, 2)))
    # model = Darknet52(input_shape=(128, 128, 2))
    # print(model.layers[-1].name)
    # print(model.layers[152].name)
    # print(model.layers[92].name)
    # print(len(model.layers))          # 185

    model = OrigUnet(input_tensor=Input((512, 512, 2)))
    model.summary()





