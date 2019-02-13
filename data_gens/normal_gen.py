import random

import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from tensorflow.python.keras.utils.np_utils import to_categorical


def augment_img_mask(image, mask, augment_param):
    horizontal_flip = 0.5 * augment_param.get('horizontal_flip', 0)
    vertical_flip = 0.5 * augment_param.get('vertical_flip', 0)
    rotation_range = [i * augment_param.get('rotation_range', 0) for i in (-1, 1)]
    brightness_range = [1 - augment_param.get('brightness_range', 0), 1 + augment_param.get('brightness_range', 0)]

    # print(horizontal_flip, vertical_flip, rotation_range, brightness_range)

    augment_img = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(horizontal_flip),
            iaa.Flipud(vertical_flip),
        ]),
        # iaa.Multiply(brightness_range),
        iaa.Affine(rotate=rotation_range)],
        # iaa.Crop(percent=(.2, .5))],
        random_order=True)

    _aug = augment_img._to_deterministic()
    image_aug = _aug.augment_image(image)
    mask_aug = _aug.augment_image(mask)
    return image_aug, mask_aug


def get_normal_generator(image_list, mask_list, batch_size=2, input_hw=(224, 224, 3), mask_hw=(224, 224, 20),
                         preprocess=True, shuffle=True, augment_param=None):
    """
    Prepare the data generator, include image pre process and image augmentation.
    :param image_list:
    :param mask_list:
    :param batch_size:
    :param input_hw:
    :param mask_hw:
    :param preprocess:
    :param shuffle:
    :param augment_param: dict, such as {'horizontal_flip':True, 'rotation_range':30}
    :return:
    """
    def norm(x):
        x = x / 127.5
        x -= 1
        return x

    assert len(image_list) == len(mask_list)

    batch_images = np.empty((batch_size, input_hw[0], input_hw[1], input_hw[2]))
    batch_masks = np.empty((batch_size, mask_hw[0], mask_hw[1], mask_hw[2]))
    batch_id = 0

    indexs = list(range(len(image_list)))
    while True:
        if shuffle:
            random.shuffle(indexs)

        for id in indexs:
            try:
                image = Image.open(image_list[id])
                image = image.resize(input_hw[0:2], Image.NEAREST)
                image_np = np.asarray(image, dtype=np.uint8)

                if preprocess:
                    image_np = norm(image_np)

                if image_np.shape != input_hw:   # try to reshape
                    image_np = np.stack((image_np,) * input_hw[2], axis=-1)

                    if image_np.shape != input_hw:
                        raise Exception('Image size wrong')

                mask = Image.open(mask_list[id])
                mask = mask.resize(mask_hw[0:2], Image.NEAREST)
                mask_np = np.asarray(mask, dtype=np.uint8).copy()

                # mask_np[mask_np != 0] = 1
                if augment_param:
                    image_np, mask_np = augment_img_mask(image_np, mask_np, augment_param)

                mask_np = to_categorical(mask_np, num_classes=mask_hw[2])

                batch_images[batch_id] = image_np
                batch_masks[batch_id] = mask_np

                batch_id += 1
                if batch_id == batch_size:
                    batch_id = 0
                    yield batch_images, batch_masks
            except FileNotFoundError:
                print('Image/mask not found, Ignore', image_list[id], mask_list[id])


