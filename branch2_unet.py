from keras.layers import Conv2D, Conv2DTranspose, Lambda, add
from keras.models import Model
from keras.optimizers import Adam
from loss import *
from backboned_unet import *
import tensorflow as tf
import os


####### custom loss #######
def bg_loss(y_true, y_pred, channel=0):
    # for bg_mask: use dice_loss
    bg_loss_ = 1 - dice_n(y_true, y_pred, channel)
    return bg_loss_


def line_loss(y_true, y_pred, channelLst=[i for i in range(1,9)]):
    # for line_mask: use dice_bce_focal_loss
    alpha = 10
    beta = 100
    line_loss_ = 0
    for i in channelLst:
        dice = dice_n(y_true, y_pred, i)
        bce = reweighting_bce_n(y_true, y_pred, i)
        focal = focal_loss_n(y_true, y_pred, i)
        line_loss_ += 1 - dice + alpha * bce + beta * focal
    return line_loss_


def cspine_loss(y_true, y_pred, channel=9):
    # for cpsine_mask: use dice_focal_loss
    alpha = 100
    cspine_loss_ = 1 - dice_n(y_true, y_pred, channel) + alpha * focal_loss_n(y_true, y_pred, channel)
    return cspine_loss_


def mixed_loss(y_true, y_pred):
    return bg_loss(y_true, y_pred, 0) + 2*line_loss(y_true, y_pred, [i for i in range(1,9)]) + 4*cspine_loss(y_true, y_pred, 9)


def test_loss(y_true, y_pred):
    return border_dice_n(y_true, y_pred, 0)


####### custom metric #######
def metric_dice_1(y_true, y_pred):
    return dice_n(y_true, y_pred, 0)
def metric_dice_2(y_true, y_pred):
    return dice_n(y_true, y_pred, 2)
def metric_dice_3(y_true, y_pred):
    return dice_n(y_true, y_pred, 8)
def metric_dice_4(y_true, y_pred):
    return dice_n(y_true, y_pred, 9)


####### custom model #######
def unet(backbone_name='orig_unet', input_shape=(256,256,1), output_channels=1, stage=5,
         lr=3e-4, decay=5e-4, freeze=False, weight_pt=''):
    # backboned encoder
    backbone, encoder_features = get_backbone(backbone_name, input_shape)
    inpt = backbone.input

    # remove average pooling layer at the end of backbone (for resnet models of certain version of keras)
    if isinstance(backbone.layers[-1], AveragePooling2D):
        x = backbone.get_layer(index=-2).output
    else:
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

    # model head
    x = Conv2D(output_channels, kernel_size=3, padding='same')(x)
    orig_model = Model(inpt, x)

    # add level2 branch
    level2 = orig_model.get_layer('activation_68').output
    level2 = Conv2D(1, 1, kernel_initializer='zeros')(level2)
    level2 = Conv2DTranspose(1, 8, strides=4, padding='same', kernel_initializer=bilinear)(level2)

    # fuse and out
    x = Lambda(element_add)([x, level2])
    x = Activation('sigmoid')(x)

    model = Model(inpt, x)
    if os.path.exists(weight_pt):
        print("load weight: ", weight_pt)
        model.load_weights(weight_pt, by_name=True, skip_mismatch=True)
    if freeze:
        for i in range(68):
            model.layers[i].trainable = False
        print("freeze the first 68 layers before level2")

    adam = Adam(lr, decay)
    # metric_lst = [metric_dice_1, metric_dice_2, metric_dice_3, metric_dice_4] + [dice_loss, bg_loss, line_loss, cspine_loss]
    model.compile(adam, loss=test_loss, metrics=[metric_dice_1])

    return model


def element_add(args):
    x, level2 = args
    level2 = tf.pad(level2, [[0,0], [0,0], [0,0], [1,2]])
    return add([x, level2])


if __name__ == '__main__':
    model = unet('orig_unet', input_shape=(512,512,2), output_channels=4, stage=5)
    # model = unet('vgg16', input_shape=(256,256,3), output_channels=1, stage=5)
    model.summary()



