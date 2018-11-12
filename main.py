# from keras import layers, models, regularizers
# import keras.backend as K
# import tensorflow as tf
# from keras.callbacks import TensorBoard, ReduceLROnPlateau,ModelCheckpoint
from PIL import Image
import os
import numpy as np
import random
from keras.utils.np_utils import to_categorical

# config
# Download VOC 2012 Dataset
voc2012_folder = 'D:\Datasets\VOC2012'

# region prepareData
print('===== prepare input/output data generator =====')


def get_voc_generator(voc_path, batch_size, input_hw=(224, 224, 3), mask_hw=(32, 32, 20), shuffle=True):
    train_set_file = os.path.join(voc_path, 'ImageSets/Segmentation/train.txt')

    batch_images = np.empty((batch_size, input_hw[0], input_hw[1], input_hw[2]))
    batch_masks = np.empty((batch_size, mask_hw[0], mask_hw[1], mask_hw[2]))
    batch_id = 0

    with open(train_set_file, 'r') as f:
        image_list = f.read().splitlines()
        while True:
            if shuffle:
                random.shuffle(image_list)

            for image_name in image_list:
                try:
                    image_path = os.path.join(voc_path, 'JPEGImages', image_name + '.jpg')
                    image = Image.open(image_path)
                    image = image.resize(input_hw[0:2], Image.NEAREST)
                    image_np = np.asarray(image, dtype=np.uint8)
                    batch_images[batch_id] = image_np

                    mask_path = os.path.join(voc_path, 'SegmentationClass', image_name + '.png')
                    mask = Image.open(mask_path)
                    mask = mask.resize(mask_hw[0:2], Image.NEAREST)
                    mask_np = np.asarray(mask, dtype=np.uint8)

                    # TODO @shutao, Do one-hot encoding, Special for Pascal VOC dataset.
                    mask_np = to_categorical(mask_np, num_classes=mask_hw[2])

                    batch_masks[batch_id] = mask_np

                    batch_id += 1
                    if batch_id == batch_size:
                        batch_id = 0
                        yield image_np, mask_np
                except FileNotFoundError as e:
                    print('Image not found', image_name)


# endregion

if __name__ == "__main__":
    gen = get_voc_generator(voc2012_folder, 3, input_hw=(224, 224, 3), mask_hw=(32, 32, 20))
    print(next(gen))
