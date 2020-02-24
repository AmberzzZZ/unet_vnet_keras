from backboned_unet import get_backbone, unet
from loss import *
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import SGD, Adam


####### custom loss #######
def disc_loss(y_true, y_pred):
    # use dice + bce (+border_dice when fine-tuning)
    alpha = 10
    beta = 100
    gamma = 0
    dice = dice_coef(y_true, y_pred)
    bce_ = reweighting_bce(y_true, y_pred)
    border_dice = border_dice_coef(y_true, y_pred)
    return 1 - dice + alpha * bce_


def tuochu_loss(y_true, y_pred):
    # use dice + bce + focal
    alpha = 10
    beta = 100
    gamma = 0
    dice = dice_coef(y_true, y_pred)
    bce_ = reweighting_bce(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    posmask = K.greater(K.sum(K.batch_flatten(y_true), axis=1, keepdims=True), 0)
    return 1 - dice + alpha * bce_ + beta * focal * (1 + 0 * tf.cast(posmask, tf.float32))


def orig_loss(y_true, y_pred):
    alpha = 1
    return disc_loss(y_true[...,0:1], y_pred[...,0:1]) + alpha * tuochu_loss(y_true[...,1:2], y_pred[...,1:2])


def roi_loss(y_true, y_pred):
    return tuochu_loss(y_true, y_pred)


####### custom metric #######
def metric_disc_dice(y_true, y_pred):
    return dice_coef(y_true[...,0:1], y_pred[...,0:1])
def metric_tuochu_dice(y_true, y_pred):
    return dice_coef(y_true[...,1:2], y_pred[...,1:2])
def metric_roi_tuochu_dice(y_true, y_pred):
    return dice_coef(y_true, y_pred)


def metric_tuochu_recall(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    y_t = K.cast(K.greater(K.sum(K.batch_flatten(y_true), axis=-1), 0), tf.int8)  # N*1
    y_1 = tf.ones_like(y_pred)
    y_0 = tf.ones_like(y_pred)
    y_bp = tf.where(tf.greater(y_pred, 0.3), y_1, y_0)
    y_p = K.cast(K.greater(K.sum(K.batch_flatten(y_bp), axis=-1), 100), tf.int8)
    return K.sum(y_t * y_p) / K.sum(y_t)


def metric_tuochu_precision(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]
    y_t = K.cast(K.greater(K.sum(K.batch_flatten(y_true), axis=-1), 0), tf.int8)  # N*1
    y_1 = tf.ones_like(y_pred)
    y_0 = tf.ones_like(y_pred)
    y_bp = tf.where(tf.greater(y_pred, 0.3), y_1, y_0)
    y_p = K.cast(K.greater(K.sum(K.batch_flatten(y_bp), axis=-1), 100), tf.int8)
    return K.sum(y_t * y_p) / K.sum(y_p)


####### custom model #######
def fine_grained_unet(backbone_name='resnet50', input_shape=(256,256,1), output_channels=[2,1], stage=5):

    unet_model = unet(backbone_name, input_shape, 1, stage)
    backbone = Model(unet_model.input, unet_model.get_layer(index=-2).output)

    full_input = Input(input_shape)
    roi_input = Input(input_shape)

    x1 = backbone(full_input)
    x2 = backbone(roi_input)

    # orig task branch
    full_output = Conv2D(output_channels[0], kernel_size=3, padding='same', activation='sigmoid',
                    use_bias=True, kernel_initializer='glorot_uniform', name='orig_branch')(x1)
    # fine-grained branch
    roi_output = Conv2D(output_channels[1], kernel_size=3, padding='same', activation='sigmoid',
                    use_bias=True, kernel_initializer='glorot_uniform', name='roi_branch')(x2)

    model = Model(inputs=[full_input, roi_input], outputs=[full_output, roi_output])

    sgd = SGD(lr=1e-4, momentum=0.97, decay=1e-6, nesterov=True)
    adam = Adam(lr=3e-4, decay=5e-6)
    metriclst = {'orig_branch': [metric_disc_dice, metric_tuochu_dice, metric_tuochu_recall, metric_tuochu_precision],
                 'roi_branch': [metric_roi_tuochu_dice, metric_roi_tuochu_recall, metric_roi_tuochu_precision]}
    model.compile(sgd, loss={'orig_branch': orig_loss, 'roi_branch': roi_loss},
                  loss_weights=[1., 1.], metrics=metriclst)

    return model


if __name__ == '__main__':
    model = fine_grained_unet(backbone_name='darknet52', input_shape=(256,256,1), output_channels=[2,1], stage=5)
    # model.summary()





