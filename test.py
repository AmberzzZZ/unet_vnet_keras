from model import *
from data import *


model = unet()
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


