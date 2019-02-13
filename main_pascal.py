import sys

from PIL import Image
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from custom_loss import *
import numpy as np
from keras.optimizers import Adam, SGD
from custom_metrics import *
from models import *
from data_gens.pascal_gen import get_voc_generator
from keras.models import load_model
import numpy as np

sys.setrecursionlimit(10000)


if __name__ == "__main__":
    # Use VOC 2012 Dataset
    voc2012_folder = 'D:\Datasets\VOC2012'
    batch_size = 8

    train_gen = get_voc_generator(voc2012_folder, 'train', batch_size, input_hw=(513, 513, 3), mask_hw=(513, 513, 21))
    val_gen = get_voc_generator(voc2012_folder, 'val', batch_size, input_hw=(513, 513, 3), mask_hw=(513, 513, 21))

    loss_weight = [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    # countinues_train_model = 'xception_deeplab_pascal.h5'
    countinues_train_model = None

    if countinues_train_model:
        model = load_model(countinues_train_model, custom_objects={'BilinearResizeLayer2D': BilinearResizeLayer2D,
                                                                'focal_loss': categorical_focal_loss(alpha=loss_weight,
                                                                                                     gamma=2.),
                                                                'mean_iou': mean_iou})
    else:
        # model = FCN.get_fcn32s_model(input_shape=(513, 513, 3), class_no=21)
        # model = Unet.get_unet_model(input_shape=(513, 513, 3), class_no=21)
        # model = DeepLabV3Plus.get_model(input_shape=(513, 513, 3), atrous_rate=(3, 6, 9), class_no=21)
        model = Xception_DeepLabV3Plus.get_model(input_shape=(513, 513, 3), atrous_rate=(3, 6, 9), class_no=21, freezeEncoder=False)
        # model = Mobilenetv2_DeepLabV3Plus.get_model(input_shape=(513, 513, 3), atrous_rate=(3, 6, 9), class_no=21, freezeEncoder=True)
        # # class 0 is the background, give it lower weight


        # model.compile(loss=weighted_categorical_crossentropy(loss_weight), optimizer='adam', metrics=[mean_iou])
        # model.compile(loss=tversky(), optimizer='adam', metrics=[mean_iou])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=[mean_iou, 'acc'])
        # model.compile(loss=categorical_focal_loss(alpha=loss_weight, gamma=2.), optimizer=Adam(lr=0.1), metrics=[mean_iou, 'acc'])

    model.compile(loss=categorical_focal_loss(alpha=loss_weight, gamma=2.), optimizer='adam', metrics=[mean_iou, 'acc'])
    model.summary()

    checkpoint = ModelCheckpoint('xception_deeplab_pascal.h5', verbose=1, save_best_only=False, period=3)
    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='mean_iou', patience=2, verbose=1, factor=0.5, mode='max', min_delta=0.002, min_lr=0.000001)

    model.fit_generator(
        train_gen,
        steps_per_epoch=200,
        epochs=200,
        validation_data=val_gen,
        validation_steps=10,
        callbacks=[checkpoint, tensor_board, learning_rate_reduction]
    )
    model.save('xception_deeplab_pascal.h5', overwrite=True, include_optimizer=False)

    print('Start Test')
    model = load_model('xception_deeplab_pascal.h5', custom_objects={'BilinearResizeLayer2D': BilinearResizeLayer2D}, compile=False)
    # 取val集10张图片，测试一下效果
    val_gen = get_voc_generator(voc2012_folder, 'val', 1, input_hw=(513, 513, 3), mask_hw=(513, 513, 21))
    # Pascal Voc 使用了indexed color, 这里提取它的palette
    mask_sample = Image.open(os.path.join(voc2012_folder, 'SegmentationClass/2007_000032.png'))
    pascal_palette = mask_sample.getpalette()

    i = 0
    for val_images, mask in val_gen:
        img_np = val_images[0]
        img_np = (img_np + 1.) * 127.5
        im0 = Image.fromarray(np.uint8(img_np))
        im0.save('output/{}_img.jpg'.format(i))

        res = model.predict(val_images)[0]
        pred_label = res.argmax(axis=2)
        im1 = Image.fromarray(np.uint8(pred_label))
        im1.putpalette(pascal_palette)
        im1.save('output/{}_pred.png'.format(i))

        true_label = mask[0].argmax(axis=2)
        im2 = Image.fromarray(np.uint8(true_label))
        im2.putpalette(pascal_palette)
        im2.save('output/{}_true.png'.format(i))

        i += 1
        if i == 100:
            print('End test')
            exit(1)



# for layer in model.layers:  # not available as pre train model is not ready here.
#     layer.trainable = True
#
# model.compile(loss=categorical_focal_loss(alpha=loss_weight, gamma=2.), optimizer=SGD(lr=0.0005),
#               metrics=[mean_iou, 'acc'])
