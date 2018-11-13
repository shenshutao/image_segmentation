# from keras import layers, models, regularizers
# import keras.backend as K
# import tensorflow as tf
# from keras.callbacks import TensorBoard, ReduceLROnPlateau,ModelCheckpoint
import os
import random

import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import load_model

from models.fcn import FCN

import sys

sys.setrecursionlimit(10000)

# config
# Download VOC 2012 Dataset
voc2012_folder = 'D:\Datasets\VOC2012'

# region prepareData
print('===== prepare input/output data generator =====')


def get_voc_generator(voc_path, train_or_val='train', batch_size=2, input_hw=(224, 224, 3), mask_hw=(32, 32, 20),
                      shuffle=True):
    if train_or_val == 'train':
        data_set_file = os.path.join(voc_path, 'ImageSets/Segmentation/train.txt')
    else:
        data_set_file = os.path.join(voc_path, 'ImageSets/Segmentation/val.txt')

    batch_images = np.empty((batch_size, input_hw[0], input_hw[1], input_hw[2]))
    batch_masks = np.empty((batch_size, mask_hw[0], mask_hw[1], mask_hw[2]))
    batch_id = 0

    with open(data_set_file, 'r') as f:
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
                    mask_np = np.asarray(mask, dtype=np.uint8).copy()

                    mask_np[mask_np == 255] = 0  # zero, indicating background.
                    mask_np = to_categorical(mask_np, num_classes=mask_hw[2])

                    batch_masks[batch_id] = mask_np

                    batch_id += 1
                    if batch_id == batch_size:
                        batch_id = 0
                        yield batch_images, batch_masks
                except FileNotFoundError:
                    print('Image not found, Ignore', image_name)


# endregion


if __name__ == "__main__":
    batch_size = 8

    train_gen = get_voc_generator(voc2012_folder, 'train', batch_size, input_hw=(224, 224, 3), mask_hw=(224, 224, 21))

    model = FCN.get_fcn32s_model(input_shape=(224, 224, 3), class_no=21)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    checkpoint = ModelCheckpoint('fcn.h5', verbose=1, save_best_only=False, period=3)
    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5, min_lr=0.000001)
    model.fit_generator(
        train_gen,
        steps_per_epoch=180,
        epochs=50,
        class_weight='auto',
        callbacks=[checkpoint, tensor_board, learning_rate_reduction]
    )

    # model.save('fcn.h5')
    #
    # model = load_model('fcn.h5')
    # 取val集10张图片，测试一下效果
    val_gen = get_voc_generator(voc2012_folder, 'val', 1, input_hw=(224, 224, 3), mask_hw=(224, 224, 21))
    # Pascal Voc 使用了indexed color, 这里提取它的palette
    mask_sample = Image.open('D:/Datasets/VOC2012/SegmentationClass/2007_000032.png')
    pascal_palette = mask_sample.getpalette()

    i = 0
    for val_images, mask in val_gen:
        res = model.predict(val_images)[0]

        pred_label = res.argmax(axis=2)

        print(pred_label)
        print(pred_label.shape)
        true_label = mask[0].argmax(axis=2)
        print(true_label)
        print(true_label.shape)

        from PIL import Image

        im0 = Image.fromarray(np.uint8(val_images[0]))
        im0.save('output/img_{}.jpg'.format(i))

        im1 = Image.fromarray(np.uint8(pred_label))
        im1.putpalette(pascal_palette)
        im1.save('output/pred_{}.png'.format(i))

        im2 = Image.fromarray(np.uint8(true_label))
        im2.putpalette(pascal_palette)
        im2.save('output/true_{}.png'.format(i))

        i += 1
        if i == 100:
            exit(1)

