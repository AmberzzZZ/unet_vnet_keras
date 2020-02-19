from backboned_unet import get_backbone, unet
from loss import *
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adam


####### custom loss #######
def disc_loss(y_true, y_pred):
    # use dice + bce (+border_dice when fine-tuning)
    alpha = 10
    beta = 100
    gamma = 0
    dice = dice_coef(y_true, y_pred)
    bce_ = bce(y_true, y_pred)
    border_dice = border_dice_coef(y_true, y_pred)
    return 1 - dice + alpha * bce_


def tuochu_loss(y_true, y_pred):
    # use dice + bce + focal
    alpha = 10
    beta = 100
    gamma = 0
    dice = dice_coef(y_true, y_pred)
    bce_ = bce(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    posmask = K.greater(K.sum(K.batch_flatten(y_true), axis=1, keepdims=True), 0)
    return 1 - dice + alpha * bce_ + beta * focal * (1 + 0 * tf.cast(posmask, tf.float32))


####### custom metric #######



####### custom model #######
def fine_grained_unet(backbone_name='resnet50', input_shape=(256,256,1), output_channels=1, stage=5):

    unet_model = unet(backbone_name, input_shape, output_channels, stage)

    full_input = Input(input_shape)
    roi_input = Input(input_shape)

    full_output = unet_model(full_input)
    roi_output = unet_model(roi_input)

    model = Model(inputs=[full_input, roi_input], outputs=[full_output, roi_output])

    sgd = SGD(lr=1e-4, momentum=0.97, decay=1e-6, nesterov=True)
    adam = Adam(lr=3e-4, decay=5e-6)
    metric_lst = {'full_output': [disc_loss, dice_coef, bce, border_dice_coef], 'roi_output': [tuochu_loss, dice_coef, bce, focal_loss]}
    model.compile(sgd, loss=[disc_loss, tuochu_loss], loss_weights=[1, 1], metrics=metric_lst)

    return model


if __name__ == '__main__':
    model = fine_grained_unet(backbone_name='darknet52', input_shape=(256,256,1), output_channels=1, stage=5)
    model.summary()



