import numpy as np
import random
import os
from PIL import Image
from keras.utils.np_utils import to_categorical

# Horse dataset, download from http://www.msri.org/people/members/eranb/
def get_horse_generator(horse_path, train_or_val='train', batch_size=2, input_hw=(224, 224, 3), mask_hw=(224, 224, 20),
                        preprocess=True, shuffle=True):
    def norm(x):
        x = x / 127.5
        x -= 1
        return x

    if train_or_val == 'train':
        image_list = ['horse{0:03d}'.format(i + 1) for i in range(250)]  # 001 to 250
    else:
        image_list = ['horse{0:03d}'.format(i + 1) for i in range(250, 328)]  # 251 to 328
        image_list.remove('horse318')  # black image in the dataset ... remove it

    batch_images = np.empty((batch_size, input_hw[0], input_hw[1], input_hw[2]))
    batch_masks = np.empty((batch_size, mask_hw[0], mask_hw[1], mask_hw[2]))
    batch_id = 0

    while True:
        if shuffle:
            random.shuffle(image_list)

        for image_name in image_list:
            try:
                image_path = os.path.join(horse_path, 'rgb', image_name + '.jpg')
                image = Image.open(image_path)
                image = image.resize(input_hw[0:2], Image.NEAREST)
                image_np = np.asarray(image, dtype=np.uint8)
                if preprocess:
                    image_np = norm(image_np)
                batch_images[batch_id] = image_np

                mask_path = os.path.join(horse_path, 'figure_ground', image_name + '.jpg')
                mask = Image.open(mask_path)
                mask = mask.resize(mask_hw[0:2], Image.NEAREST)
                mask_np = np.asarray(mask, dtype=np.uint8).copy()

                mask_np[mask_np != 0] = 1  # zero, indicating horse.
                mask_np = to_categorical(mask_np, num_classes=mask_hw[2])

                batch_masks[batch_id] = mask_np

                batch_id += 1
                if batch_id == batch_size:
                    batch_id = 0
                    yield batch_images, batch_masks
            except FileNotFoundError:
                print('Image not found, Ignore', image_name)


# np.set_printoptions(threshold=np.nan)
# horse_path = 'C:/Users/AlphaCat/Desktop/image_segmentation/weizmann_horse_db'
# horse_gen = get_horse_generator(horse_path, train_or_val='train', batch_size=2, input_hw=(32, 32, 3),
#                                 mask_hw=(32, 32, 2),
#                                 preprocess=True, shuffle=True)
#
# print(next(horse_gen))
# print(next(horse_gen)[0].shape)
# print(next(horse_gen)[1].shape)
