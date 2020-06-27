from fine_grained_unet import *
from dataLoader import trainGeneratorFineGrained
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


if __name__ == '__main__':

    batch_size = 6
    train_path = "data/train"
    val_path = "data/test"
    image_folder = "image"
    mask_folder = "label"
    multi_class = True
    output_channels = [2, 1]
    data_gen_args = None

    # data
    train_generator = trainGeneratorFineGrained(train_path, image_folder, mask_folder, data_gen_args, batch_size, multi_class)
    val_generator = trainGeneratorFineGrained(val_path, image_folder, mask_folder, data_gen_args, batch_size, multi_class)

    # model
    model = fine_grained_unet(backbone_name='darknet52', input_shape=(512,512,1), output_channels=output_channels)
    model.load_weights("yolov3.h5", by_name=True, skip_mismatch=True)

    # callbacks
    weight_file_pt = 'fg_darknet_{epoch:02d}_{val_loss:.3f}_{val_orig_branch_metric_disc_dice:.3f}_{val_orig_branch_metric_tuochu_dice:.3f}_{val_roi_branch_metric_roi_tuochu_dice:.3f}.hdf5'
    checkpoint = ModelCheckpoint(weight_file_pt, monitor='val_loss', verbose=1, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # train
    steps_per_epoch = len(os.listdir(train_path+image_folder)) // batch_size
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=100,
                        verbose=1,
                        validation_data=val_generator,
                        validation_steps=steps_per_epoch//6,
                        callbacks=[checkpoint],
                        # callbacks=[checkpoint, reduce_lr, early_stopping],
                        )



