from keras.models import *
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras import backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1 - pt_1, gamma) * K.log(pt_1)) -   \
        K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1 - pt_0))


def bce(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(K.log(pt_1)) - K.mean(K.log(1.-pt_0))
    return score


def reweighting_bce(y_true, y_pred):
    alpha = 0.999
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    score = -K.mean(alpha * K.log(pt_1)) - K.mean((1-alpha) * K.log(1.-pt_0))
    # beta = tf.constant([0.02, 0.49, 0.49])
    # score = -K.mean(alpha * K.log(pt_1), axis=(0,1,2)) - K.mean((1-alpha) * K.log(1.-pt_0), axis=(0,1,2))
    # score = K.sum(beta * score)
    return score


def mixed_loss(y_true, y_pred):
    alpha = 100
    focal_dice = alpha * focal_loss(y_true, y_pred) - K.log(1 + dice_loss(y_true, y_pred))
    bce_dice = alpha * reweighting_bce(y_true, y_pred) + dice_loss(y_true, y_pred)
    return bce_dice


def crop_margin(conv, up):
    w1, h1 = conv.shape[1], conv.shape[2]
    w2, h2 = up.shape[1], up.shape[2]
    gap1, gap2, gap3, gap4 = map(int, ((w1-w2)//2, w1-w2-(w1-w2)//2, (h1-h2)//2, h1-h2-(h1-h2)//2))
    return ((int(gap1), int(gap2)), (int(gap3), int(gap4)))


def unet_original(input_size=(256,256,1), output_channels=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5))
    conv4 = Cropping2D(cropping=crop_margin(conv4, up6))(conv4)
    merge6 = concatenate([conv4,up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv6))
    conv3 = Cropping2D(cropping=crop_margin(conv3, up7))(conv3)
    merge7 = concatenate([conv3,up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv2 = Cropping2D(cropping=crop_margin(conv2, up8))(conv2)
    merge8 = concatenate([conv2,up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv8))
    conv1 = Cropping2D(cropping=crop_margin(conv1, up9))(conv1)
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    sgd = SGD(lr=1e-4, momentum=0.9, decay=0., nesterov=True)
    adam = Adam(lr=1e-5)
    model.compile(optimizer=sgd, loss=dice_loss, metrics=['accuracy', dice_coef])

    return model


def conv_block(x, n_filters, kernel_size=3, strides=1, padding='same', activation='relu', res=False):
    inpt = x
    x = Conv2D(n_filters, kernel_size, padding=padding, strides=strides, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, kernel_size, padding=padding, strides=strides, activation=activation)(x)
    x = BatchNormalization()(x)
    return Concatenate()([inpt, x]) if res else x


def comp_block(x, stage, res=False):   # [0,1,2,3,4]
    x = conv_block(x, 64*(2**stage))
    feature = x
    if stage < 4:
        x = MaxPool2D()(x)
    return x, feature


def up_block(x, shortcut, stage, upSam=True):      # [4, 3, 2, 1]
    # up
    if upSam:
        x = UpSampling2D()(x)
        x = Conv2D(64*(2**(stage-1)), 2, padding='same', activation='relu')(x)
    else:
        x = Conv2DTranspose(64*(2**(stage-1)), 3, strides=2, activation=activation, padding='same')(x)
    # concantenate
    x = concatenate([x,shortcut], axis=3)
    # conv
    x = conv_block(x, 64*(2**(stage-1)))
    return x


def unet_padding(input_size=(256,256,1), output_channels=1, stage=5):
    inpt = Input(input_size)

    # compression
    x = inpt
    features = []
    for i in range(stage):    # [0,1,2,3,4]
        x, feature = comp_block(x, i)
        features.append(feature)

    # decompression
    for i in range(stage-1, 0, -1):    # [4,3,2,1]
        x = up_block(x, features[i-1], i, upSam=True)

    # output
    x = Conv2D(output_channels, 1, activation='sigmoid')(x)

    model = Model(inputs=inpt, outputs=x)

    sgd = SGD(lr=1e-4, momentum=0.9, decay=0., nesterov=True)
    model.compile(optimizer=sgd, loss=dice_loss, metrics=[dice_coef, reweighting_bce, bce])

    return model


if __name__ == '__main__':
    # model = unet_original(input_size=(448,448,1), output_channels=1)
    model = unet_padding(input_size=(256,256,1), output_channels=4)
    model.summary()



