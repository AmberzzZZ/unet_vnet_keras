from vnet import *
import cv2
import numpy as np


if __name__ == '__main__':

    model = vnet(input_size=(256,256,1))
    model.load_weights("vnet_membrane_06_dice_0.921.h5", by_name=True)

    test_img = cv2.imread("data/membrane/test/0.png", 0)
    tmp = cv2.resize(test_img, (256, 256))
    tmp = tmp / 255.
    tmp = np.reshape(tmp, (1,256,256,1))
    mask = model.predict(tmp)
    print(np.min(mask), np.max(mask))

    # postprocess
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0
    cv2.imshow("mask", mask[0,:,:,0])
    cv2.waitKey(0)


