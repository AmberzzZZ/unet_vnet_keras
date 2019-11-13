from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import cv2


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):     # reflect each class into one channel of the mask
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,
                   target_size = (256,256),flag_multi_class = False,num_class = 2,
                   image_color_mode = "grayscale",mask_color_mode = "grayscale",
                   image_save_prefix  = "image",mask_save_prefix  = "mask",
                   save_to_dir = None,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path,num_image=30,target_size=(256,256),  \
                  flag_multi_class = False,as_gray = True):
    cnt = 0
    for test_img in [i for i in os.listdir(test_path) if i[-3:]=='jpg' or i[-3:]=='png']:
        if cnt >= num_image:
            break
        cnt += 1
        img = cv2.imread(os.path.join(test_path,test_img), flags=(0 if as_gray else 1))
        if np.max(img) > 1:
            img = img / 255
        img = cv2.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.expand_dims(img, axis=0)
        yield img


def geneTrainNpy(image_path,mask_path,
                 flag_multi_class = False,num_class = 2,
                 image_prefix = "image",mask_prefix = "mask",
                 image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.imread(item, flags=(0 if image_as_gray else 1))
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix), flags=(0 if image_as_gray else 1))
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def geneTestNpy(image_path,
                flag_multi_class = False,num_class = 2,
                image_prefix = "image",
                image_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.imread(item, flags=(0 if image_as_gray else 1))
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        if(np.max(img) > 1):
            img = img / 255
        image_arr.append(img)
    image_arr = np.array(image_arr)
    return image_arr

