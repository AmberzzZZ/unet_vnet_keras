from fine_grained_unet import *
import numpy as np
import cv2
import os
import glob
import random
import keras.backend as K


if __name__ == '__main__':

    batch_size = 6
    test_path = "data/test/image"
    test_label_path = "data/test/label"
    des = "result"
    output_channels = [2, 1]

    model = fine_grained_unet('darknet52', input_shape=(512,512,1), output_channels=output_channels)
    model.load_weights("fg_darknet*.hdf5")

    tp = tn = fp = fn = 0
    for pngfile in [i for i in glob.glob(test_path + "/*png")][:]:
        img = cv2.imread(pngfile, 0)
        if np.max(img) > 1:
            img = img / 255.

        # orig_task branch do first
        full_input = np.reshape(img, (1, 512, 512, 1))
        func1 = K.function(inputs=[model.inputs[0]],
                           outputs=[model.get_layer('orig_branch').output])
        y1 = func1([full_input])[0]
        y1[y1>=0.3] = 1
        y1[y1<0.3] = 0

        # roi branch do next
        disc_mask = y1[0,:,:,0]
        _, contours, hierarchy = cv2.findContours(disc_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        coords = contours[0].reshape((-1,2))
        x_min, y_min, x_max, y_max = np.min(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,0]), np.max(coords[:,1])
        x_center, y_center = (x_max + x_min) // 2, (y_max + y_min) // 2
        halfsize = max(y_max - y_center, x_max - x_center) + 20
        cropped_img = img[y_center-halfsize:y_center+halfsize, x_center-halfsize:x_center+halfsize]
        cropped_img = cv2.reshape(cropped_img, (512,512))
        roi_input = np.reshape(cropped_img, (1, 512, 512, 1))
        func2 = K.function(inputs=[model.inputs[1]],
                           outputs=[model.get_layer('roi_branch').output])
        y2 = func2([roi_input])[0]
        y2[y2>=0.3] = 1
        y2[y2<0.3] = 0
        scaled_back_y2 = np.zeros_like(y2)
        scaled_back_y2[0,y_center-halfsize:y_center+halfsize,x_center-halfsize:x_center+halfsize,0] = cv2.reshape(y2[0,:,:,0], (halfsize*2, halfsize*2))

        # visualize
        tmp = np.zeros((512,512,3))
        tmp[..., 0][y1[0,:,:,0]>0] = 255    # blue
        tmp[..., 1][y1[0,:,:,1]>0] = 255    # green
        tmp[..., 2][scaled_back_y2[0,:,:,0]>0] = 255        # red
        alpha = 0.5
        beta = 1. - alpha
        gamma = 0.
        img = cv2.imread(pngfile, 3)
        if np.max(img) <= 1:
            img = np.uint8(img * 255)
        img_add = cv2.addWeighted(img, alpha, tmp, beta, gamma, dtype=cv2.CV_8UC3)

        filename = pngfile.split("/")[-1]
        cv2.imwrite(os.path.join(des, filename), img_add)

        # evaluate
        gtfile = cv2.imread(os.path.join(test_label_path, filename), 0)
        y = y1[0,:,:,1]     # y2[0,:,:,0]
        if np.sum(y) > 10:      # positive
            if 50 in gtfile:    # true
                tp += 1
            else:
                fp += 1
        else:     # negative
            if 50 in gtfile:    # false
                fn += 1
            else:
                tn += 1

    print("tp: ", tp, " fp: ", fp, " fn: ", fn, " tn: ", tn)
    print("recall: ", tp / float(tp+fn))
    print("precision: ", tp / float(tp+fp))
    print("accuracy: ", (tp+tn) / float(tp+tn+fp+fn))






