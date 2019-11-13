from model import *
from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(20,'data/disc/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_disc_6layer.hdf5', monitor='loss',verbose=1, save_best_only=True)


# train with real-time img generator
model.fit_generator(myGene,steps_per_epoch=3,epochs=10,callbacks=[model_checkpoint])
# steps_per_epoch: 120/2

# # train with npy file
# imgs_train = np.load("data/image_arr.npy")
# imgs_mask_train = np.load("data/mask_arr.npy")
# model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])