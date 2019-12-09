from models import *
from data import *
import cv2
import numpy as np


# ### for original model ###
# # assume we want to predict a image of 256*256
# # for a 448*448 input, the model gives 260*260 outputs
# model = unet_original(input_size=(448,448,1), output_channels=1)
# model.load_weights("filepath", by_name=True, skip_mismatch=True)
# test_file = "data/disc/test/n/1.2.840.113619.2.289.3.676096.21.1540685771.647.4_L4L5_5.jpg"
# img = cv2.imread(test_file, 0)
# img = cv2.resize(img, (256, 256))
# w, h = img.shape
# img = cv2.copyMakeBorder(img, (448-h)//2, 448-h-(448-h)//2, (448-w)//2, 448-w-(448-w)//2, cv2.BORDER_REFLECT)  #   BORDER_REFLECT
# img = np.expand_dims(img, axis=0)
# img = np.expand_dims(img, axis=-1)
# pred = model.predict(img)
# pw, ph = pred.shape
# pred = pred[0,ph//2-h//2:ph//2-h//2+h,pw//2-w//2:pw//2-w//2+w,0]


### for padding model ###
# for a 256*256 input, the model gives 256*256 prediction directly
model = unet_padding()
model.load_weights("unet_disc.hdf5")

# testGene = testGenerator("data/membrane/test/")
# results = model.predict_generator(testGene,1,verbose=1)     # np array (n_imgs, 256, 256, 1)
# saveResult("data/disc/test",results)

img_arr = geneTestNpy("data/disc/train/aug/")
print(img_arr.shape)
mask = model.predict(img_arr[:1])


# postprocess
mask[mask>=0.5] = 1
mask[mask<0.5] = 0
for i, item in enumerate(mask):
    cv2.imshow("mask", item[:,:,0])
    cv2.waitKey(0)


