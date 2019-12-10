from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import random


# using ImageDataGenerator: work only for single channel
def trainGenerator(train_path, image_folder, mask_folder,
                   aug_dict, batch_size, target_size=(256,256), seed=1):
    datagen = ImageDataGenerator(**aug_dict)
    image_generator = datagen.flow_from_directory(train_path,
                                                  classes=[image_folder],
                                                  class_mode=None,
                                                  color_mode='grayscale',
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  seed=seed)
    mask_generator = datagen.flow_from_directory(train_path,
                                                 classes=[mask_folder],
                                                 class_mode=None,
                                                 color_mode='grayscale',
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        if np.max(img) > 1:
            img /= 255
        if np.max(mask) > 1:
            mask[mask>0] = 1
        yield (img,mask)


# using self define function: read in npy for multi-channel mask & png for single-channel mask
# to be added: online augmentation
def trainGenerator2(train_path, image_folder, mask_folder, aug_dict,
                    batch_size, multi_class=False):
    filelst = os.listdir(os.path.join(train_path, image_folder))
    idx = [i for i in range(len(filelst))]
    while 1:
        random.shuffle(idx)
        img_batch = []
        mask_batch = []
        for i in idx:
            pngfile = filelst[i]
            img = cv2.imread(os.path.join(train_path, image_folder, pngfile), 0)
            # img = np.load(os.path.join(train_path, image_folder, npyfile))
            if multi_class:
                npyfile = pngfile[:-3] + 'npy'
                mask = np.load(os.path.join(train_path, mask_folder, npyfile))
            else:
                mask = cv2.imread(os.path.join(train_path, mask_folder, pngfile), 0)
            if img is None or mask is None:
                continue
            # img, mask = augmentation(img, mask, aug_dict)
            if np.max(img) > 1:
                img = img / 255.
            if np.max(mask) > 1:
                mask[mask>0] = 1
            img_batch.append(img)
            mask_batch.append(mask)
            if len(img_batch) == batch_size:
                break

        img_batch, mask_batch = np.array(img_batch), np.array(mask_batch)
        if len(img_batch.shape) < 4:
            img_batch = np.expand_dims(img_batch, axis=-1)
        if len(mask_batch.shape) < 4:
            mask_batch = np.expand_dims(mask_batch, axis=-1)
        yield img_batch, mask_batch


if __name__ == '__main__':
    batch_size = 8
    train_path = "data/membrane/train/"
    image_folder = "image"
    mask_folder = "label"
    target_size = (512, 512)

    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         # shear_range=0.05,
                         # zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='reflect')
    # trainGenerator
    # train_generator = trainGenerator(train_path, image_folder, mask_folder, data_gen_args, batch_size, target_size)

    # trainGenerator2
    train_generator = trainGenerator2(train_path, image_folder, mask_folder, data_gen_args, batch_size)

    for idx, (img, mask) in enumerate(train_generator):
        print(img.shape, np.max(img), np.min(img))
        print(mask.shape)

        if idx > 5:
            break



