from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, Conv2DTranspose, Add
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


def unet(backbone_name='resnet50', input_shape=(256,256,1), output_channels=1, stage=5):
    # backboned encoder
    backbone, encoder_features = get_backbone(backbone_name, input_shape)
    backbone.summary()
    inpt = backbone.input
    x = backbone.output

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], MaxPooling2D):
        x = Conv3x3BnReLU(x, 512)
        x = Conv3x3BnReLU(x, 512)

     # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in encoder_features])

    # building decoder blocks
    decoder_filters=(256, 128, 64, 32, 16)
    # decoder_filters=(2048, 1024, 512, 256, 64)
    for i in range(stage):     # [0,1,2,3,4]
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        x = decoder_block_deconv(x, skip, decoder_filters[i])

    # # model head
    x = Conv2D(output_channels, kernel_size=3, padding='same', activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')(x)

    model = Model(inpt, x)
    return model


def get_backbone(backbone_name, input_shape):
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling=None)
    models = {'vgg16': vgg16, 'resnet50': resnet50}
    encoder_features = {'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
                        'resnet50': ('activation_40', 'activation_22', 'activation_10', 'activation_1')}
    return models[backbone_name], encoder_features[backbone_name]


def Conv3x3BnReLU(x, n_filters, padding='same', strides=1, activation='relu'):
    x = Conv2D(n_filters, kernel_size=3, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def Conv1x1BnReLU(x, n_filters, padding='same', strides=1, activation='relu'):
    x = Conv2D(n_filters, kernel_size=1, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def decoder_block_up(x, shortcut, n_filters):
    # upsampling
    input_filters = x.shape.as_list()[-1]
    output_filters = shortcut.shape.as_list()[-1] if shortcut is not None else n_filters
    x = Conv1x1BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = UpSampling2D()(x)
    x = Conv3x3BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = Conv1x1BnReLU(x, output_filters, padding='same', strides=1, activation='relu')
    # add
    if shortcut is not None:
        x = Add()([x, shortcut])
    return x


def decoder_block_deconv(x, shortcut, n_filters):
    # convTranspose
    input_filters = x.shape.as_list()[-1]
    output_filters = shortcut.shape.as_list()[-1] if shortcut is not None else n_filters
    x = Conv1x1BnReLU(x, input_filters//4, padding='same', strides=1, activation='relu')
    x = Conv2DTranspose(input_filters//4, kernel_size=4, padding='same', strides=2)(x)
    x = Conv1x1BnReLU(x, output_filters, padding='same', strides=1, activation='relu')
    # add
    if shortcut is not None:
        x = Add()([x, shortcut])
    return x


if __name__ == '__main__':
    model = unet('resnet50', input_shape=(800,448,3), output_channels=1, stage=5)
    # model = unet('vgg16', input_shape=(256,256,3), output_channels=1, stage=5)
    model.summary()



