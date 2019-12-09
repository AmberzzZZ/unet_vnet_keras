from model import *
from data import *
from keras.callbacks import ModelCheckpoint


# model
model = unet()
# load pretrained
model.load_weights("unet_disc.hdf5", by_name=True, skip_mismatch=True)

# checkpoint
filepath = "unet_disc_{epoch:02d}_dice_{dice_coef:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, mode='min')

# train with real-time img generator
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(4,'data/disc/train','image','label',data_gen_args,save_to_dir=None)
model.fit_generator(myGene,steps_per_epoch=3,
                    epochs=10,
                    callbacks=[checkpoint])

# # train with npy file
# image_arr,mask_arr = geneTrainNpy("data/disc/train/aug/","data/disc/train/aug/")
# np.save("data/image_arr.npy",image_arr)
# np.save("data/mask_arr.npy",mask_arr)
# imgs_train = np.load("data/image_arr.npy")
# imgs_mask_train = np.load("data/mask_arr.npy")
# model.fit(imgs_train, imgs_mask_train, batch_size=2,
#           nb_epoch=10, verbose=1,
#           validation_split=0.2,
#           shuffle=True,
#           callbacks=[checkpoint])



