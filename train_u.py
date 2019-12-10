# train unet
from unet import *
from dataLoader import *
from keras.callbacks import ModelCheckpoint


if __name__ == '__main__':

    batch_size = 4
    train_path = "data/membrane/train/"
    image_folder = "image"
    mask_folder = "label"
    target_size = (256, 256)

    # train with real-time img generator
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         # shear_range=0.05,
                         # zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='reflect')
    train_generator = trainGenerator(train_path, image_folder, mask_folder, data_gen_args, batch_size, target_size)

    # model
    model = unet_padding(input_size=(256,256,1))
    model.load_weights("unet_membrane.hdf5", by_name=True, skip_mismatch=True)

    # checkpoint
    filepath = "unet_membrane_{epoch:02d}_dice_{dice_coef:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='dice_loss',verbose=1, save_best_only=True, mode='min')

    # train
    model.fit_generator(train_generator,
                        steps_per_epoch=8,
                        epochs=10,
                        callbacks=[checkpoint])
